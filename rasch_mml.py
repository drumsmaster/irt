"""
Rasch 1PL marginal maximum-likelihood calibration.

The implementation follows the Bock-Aitkin EM construction:

1. E-step: distribute each person's response pattern over a grid of latent
   ability points using posterior probabilities.
2. M-step for items: estimate every Rasch difficulty from the resulting
   expected counts.
3. M-step for the latent distribution:
      latent="normal"    -> estimate a discretized normal distribution;
      latent="empirical" -> estimate a free probability at every grid point.

The empirical option is fixed-grid nonparametric MML.  It intentionally has
no smoothing penalty.  The fitted distribution can therefore look irregular;
that is not a convergence failure.  Use a reasonably dense grid and inspect
edge mass to ensure that theta_range is wide enough.

References
----------
Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
item parameters: Application of an EM algorithm. Psychometrika, 46, 443-459.

Mislevy, R. J. (1984). Estimating latent distributions. Psychometrika, 49,
359-381.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


# ============================================================
# Data packing
# ============================================================

class MMLData:
    """Pack sparse person-item responses, grouping persons by test length."""

    def __init__(self, persons, item_list, min_person_responses=1):
        self.item_list = list(item_list)
        self.idx = {item_id: j for j, item_id in enumerate(self.item_list)}
        self.J = len(self.item_list)

        groups = {}
        for person_id, responses in persons.items():
            item_indices, values = [], []
            for item_id, value in responses.items():
                if item_id in self.idx:
                    item_indices.append(self.idx[item_id])
                    values.append(int(value))
            if len(item_indices) < min_person_responses:
                continue
            group = groups.setdefault(len(item_indices), ([], []))
            group[0].append(item_indices)
            group[1].append(values)

        self.groups = [
            (K, np.asarray(I, dtype=np.int64), np.asarray(X, dtype=np.float64))
            for K, (I, X) in sorted(groups.items())
        ]
        self.n_persons = sum(I.shape[0] for _, I, _ in self.groups)
        self.n_responses = sum(I.size for _, I, _ in self.groups)


def filter_bank(persons, items, min_item_responses=30):
    """Remove items with too few responses or no response variation."""
    n_responses = defaultdict(int)
    n_correct = defaultdict(int)

    for responses in persons.values():
        for item_id, value in responses.items():
            if item_id in items:
                n_responses[item_id] += 1
                n_correct[item_id] += int(value)

    keep = set()
    bad_items = {}
    for item_id in items:
        n = n_responses.get(item_id, 0)
        c = n_correct.get(item_id, 0)
        if n < min_item_responses:
            bad_items[item_id] = {
                "reason": "too_few_responses",
                "totalResponses": n,
            }
        elif c == 0:
            bad_items[item_id] = {
                "reason": "all_incorrect",
                "totalResponses": n,
            }
        elif c == n:
            bad_items[item_id] = {
                "reason": "all_correct",
                "totalResponses": n,
            }
        else:
            keep.add(item_id)

    filtered = {
        person_id: {
            item_id: int(value)
            for item_id, value in responses.items()
            if item_id in keep
        }
        for person_id, responses in persons.items()
    }
    filtered = {person_id: r for person_id, r in filtered.items() if r}

    return (
        filtered,
        sorted(keep),
        bad_items,
        dict(n_responses),
        dict(n_correct),
    )


# ============================================================
# Starting values only
# ============================================================

def jml_start(data: MMLData, sweeps=30, clip=8.0):
    """Short JML run used only to obtain non-degenerate starting values."""
    J = data.J
    b = np.zeros(J)
    thetas = [np.zeros(I.shape[0]) for _, I, _ in data.groups]

    for _ in range(sweeps):
        for group_index, (_, I, X) in enumerate(data.groups):
            theta = thetas[group_index]
            for _ in range(4):
                P = expit(theta[:, None] - b[I])
                score = (X - P).sum(axis=1)
                info = np.maximum((P * (1.0 - P)).sum(axis=1), 1e-8)
                theta = np.clip(theta + score / info, -2.0 * clip, 2.0 * clip)
            thetas[group_index] = theta

        score = np.zeros(J)
        info = np.zeros(J)
        for group_index, (_, I, X) in enumerate(data.groups):
            P = expit(thetas[group_index][:, None] - b[I])
            score += np.bincount(
                I.ravel(), weights=(P - X).ravel(), minlength=J
            )
            info += np.bincount(
                I.ravel(), weights=(P * (1.0 - P)).ravel(), minlength=J
            )
        b = np.clip(b + score / np.maximum(info, 1e-8), -2.0 * clip, 2.0 * clip)
        b -= b.mean()

    all_theta = np.concatenate(thetas) if thetas else np.zeros(1)
    return b, float(all_theta.mean()), float(np.clip(all_theta.std(), 0.5, clip))


# ============================================================
# Latent distributions
# ============================================================

def _normal_log_weights(grid, mu, sigma):
    sigma = max(float(sigma), 1e-8)
    logw = -0.5 * ((grid - mu) / sigma) ** 2
    return logw - logsumexp(logw)


def _normal_weights(grid, mu, sigma):
    return np.exp(_normal_log_weights(grid, mu, sigma))


def _fit_discrete_normal(post_sum, grid):
    """Exact M-step for a normal distribution discretized on `grid`."""
    N = float(post_sum.sum())
    spacing = float(np.median(np.diff(grid))) if len(grid) > 1 else 1.0
    sigma_min = max(abs(spacing) / 4.0, 0.05)
    sigma_max = max(float(np.ptp(grid)), sigma_min * 2.0)

    # Posterior moments are a good starting point for the two-parameter solve.
    mu_moment = float(np.sum(post_sum * grid) / N)
    var_moment = float(np.sum(post_sum * (grid - mu_moment) ** 2) / N)
    x0 = np.array([
        np.clip(mu_moment, grid.min(), grid.max()),
        np.log(np.clip(np.sqrt(max(var_moment, sigma_min ** 2)),
                       sigma_min, sigma_max)),
    ])

    def objective_and_gradient(x):
        mu, log_sigma = float(x[0]), float(x[1])
        sigma = np.exp(log_sigma)
        logw = _normal_log_weights(grid, mu, sigma)
        w = np.exp(logw)
        residual = post_sum - N * w
        z = (grid - mu) / sigma
        value = -float(np.dot(post_sum, logw))
        gradient = -np.array([
            np.dot(residual, z / sigma),
            np.dot(residual, z ** 2),
        ])
        return value, gradient

    result = minimize(
        objective_and_gradient,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=[
            (float(grid.min()), float(grid.max())),
            (float(np.log(sigma_min)), float(np.log(sigma_max))),
        ],
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 200},
    )

    mu = float(result.x[0])
    sigma = float(np.exp(result.x[1]))
    return mu, sigma, _normal_weights(grid, mu, sigma), bool(result.success)


# ============================================================
# EM steps
# ============================================================

def _expectation(b, grid, weights, data: MMLData, chunk=2000):
    """E-step: posterior probabilities and expected item-by-node counts."""
    J, Q = data.J, len(grid)
    log_weights = np.log(np.maximum(weights, 1e-300))

    z = grid[None, :] - b[:, None]
    logP = -np.logaddexp(0.0, -z)
    logQ = -np.logaddexp(0.0, z)

    loglik = 0.0
    n_jq = np.zeros((J, Q))
    r_jq = np.zeros((J, Q))
    post_sum = np.zeros(Q)

    for K, I, X in data.groups:
        for start in range(0, I.shape[0], chunk):
            end = min(start + chunk, I.shape[0])
            Ic = I[start:end]
            Xc = X[start:end]
            n = Ic.shape[0]

            log_response = np.zeros((n, Q))
            for k in range(K):
                item_index = Ic[:, k]
                log_response += np.where(
                    Xc[:, k][:, None] > 0.5,
                    logP[item_index],
                    logQ[item_index],
                )

            log_joint = log_response + log_weights[None, :]
            log_denominator = logsumexp(log_joint, axis=1)
            posterior = np.exp(log_joint - log_denominator[:, None])

            loglik += float(log_denominator.sum())
            post_sum += posterior.sum(axis=0)

            # A[item, person] = 1 when that person received the item.
            rows = Ic.ravel()
            cols = np.repeat(np.arange(n), K)
            administered = sp.csr_matrix(
                (np.ones(rows.size), (rows, cols)), shape=(J, n)
            )
            correct = sp.csr_matrix(
                (Xc.ravel(), (rows, cols)), shape=(J, n)
            )
            n_jq += administered @ posterior
            r_jq += correct @ posterior

    return loglik, n_jq, r_jq, post_sum


def _update_items(b, grid, n_jq, r_jq, max_newton=50):
    """M-step for Rasch item difficulties using expected counts."""
    r_total = r_jq.sum(axis=1)
    updated = b.copy()

    for _ in range(max_newton):
        P = expit(grid[None, :] - updated[:, None])
        score = (n_jq * P).sum(axis=1) - r_total
        info = np.maximum((n_jq * P * (1.0 - P)).sum(axis=1), 1e-12)
        step = np.clip(score / info, -1.0, 1.0)
        updated += step
        if float(np.max(np.abs(step))) < 1e-10:
            break

    return updated


def _center_items_and_grid(b, grid, latent, mu):
    """Set mean(b)=0 without changing any theta-b differences."""
    shift = float(b.mean())
    b = b - shift
    grid = grid - shift
    if latent == "normal":
        mu = float(mu - shift)
    return b, grid, mu


# ============================================================
# Calibration
# ============================================================

def calibrate_mml(
    persons,
    items,
    min_item_responses=30,
    latent="normal",
    grid=None,
    n_nodes=97,
    theta_range=(-14.0, 14.0),
    itemsInitialGuess=None,
    warm_start=True,
    jml_sweeps=30,
    max_iter=1000,
    loglik_tol=1e-6,
    item_tol=1e-4,
    latent_tol=1e-3,
    chunk=2000,
    verbose=True,
):
    """Calibrate Rasch item difficulties by marginal maximum likelihood.

    Parameters
    ----------
    latent : {"normal", "empirical"}
        "normal" estimates a discretized normal latent distribution.
        "empirical" estimates one free probability for every grid point.
    theta_range, n_nodes :
        Numerical integration grid.  For the empirical option this is also
        the support of the estimated mixing distribution.  Increase the range
        if the reported edge mass is not negligible.
    loglik_tol :
        Relative change in marginal log likelihood required for convergence.
    item_tol :
        Maximum absolute change in any item difficulty.
    latent_tol :
        For empirical MML, total-variation change in the weights.  For normal
        MML, the larger of the changes in mu and log(sigma).

    Returns
    -------
    itemsParams, badItems, info, status
    """
    if latent not in {"normal", "empirical"}:
        raise ValueError("latent must be 'normal' or 'empirical'")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    persons_f, item_list, bad_items, n_responses, n_correct = filter_bank(
        persons, items, min_item_responses=min_item_responses
    )
    if len(item_list) < 3:
        raise ValueError("MML: fewer than 3 items survive filtering")

    data = MMLData(persons_f, item_list)
    if data.n_persons == 0:
        raise ValueError("MML: no persons have responses to retained items")

    if grid is None:
        grid = np.linspace(theta_range[0], theta_range[1], n_nodes)
    else:
        grid = np.asarray(grid, dtype=float).copy()
        if grid.ndim != 1 or len(grid) < 5 or np.any(np.diff(grid) <= 0):
            raise ValueError("grid must be a strictly increasing 1-D array")

    if warm_start:
        b, mu, sigma = jml_start(data, sweeps=jml_sweeps)
    else:
        b = np.zeros(data.J)
        mu, sigma = 0.0, 2.0

    # Apply any supplied item starting values after constructing the retained
    # item order. Missing items keep their JML starting value.
    if itemsInitialGuess:
        for j, item_id in enumerate(item_list):
            if item_id in itemsInitialGuess and "b" in itemsInitialGuess[item_id]:
                b[j] = float(itemsInitialGuess[item_id]["b"])
        b -= b.mean()

    if latent == "normal":
        weights = _normal_weights(grid, mu, sigma)
    else:
        weights = _normal_weights(grid, mu, sigma)

    b, grid, mu = _center_items_and_grid(b, grid, latent, mu)

    loglik, n_jq, r_jq, post_sum = _expectation(
        b, grid, weights, data, chunk=chunk
    )
    converged = False
    normal_mstep_ok = True
    last_loglik_change = np.inf
    last_item_change = np.inf
    last_latent_change = np.inf

    if verbose:
        print(
            f"[MML] start | latent={latent} | items={data.J} | "
            f"persons={data.n_persons} | responses={data.n_responses}"
        )

    for iteration in range(1, max_iter + 1):
        old_b = b.copy()
        old_weights = weights.copy()
        old_mu, old_sigma = mu, sigma

        new_b = _update_items(b, grid, n_jq, r_jq)

        if latent == "empirical":
            new_weights = np.maximum(post_sum / data.n_persons, 1e-300)
            new_weights /= new_weights.sum()
            new_mu, new_sigma = mu, sigma
        else:
            new_mu, new_sigma, new_weights, normal_mstep_ok = _fit_discrete_normal(
                post_sum, grid
            )

        new_b, new_grid, new_mu = _center_items_and_grid(
            new_b, grid.copy(), latent, new_mu
        )

        new_loglik, new_n_jq, new_r_jq, new_post_sum = _expectation(
            new_b, new_grid, new_weights, data, chunk=chunk
        )

        last_loglik_change = float(new_loglik - loglik)
        last_item_change = float(np.max(np.abs(new_b - old_b)))
        if latent == "empirical":
            last_latent_change = float(0.5 * np.sum(np.abs(new_weights - old_weights)))
        else:
            last_latent_change = float(max(
                abs(new_mu - old_mu),
                abs(np.log(new_sigma) - np.log(old_sigma)),
            ))

        if last_loglik_change < -1e-6 * (1.0 + abs(loglik)):
            raise RuntimeError(
                "MML log likelihood decreased materially; this indicates a "
                "numerical or implementation problem"
            )

        b, grid, weights = new_b, new_grid, new_weights
        mu, sigma = new_mu, new_sigma
        loglik, n_jq, r_jq, post_sum = (
            new_loglik,
            new_n_jq,
            new_r_jq,
            new_post_sum,
        )

        relative_loglik_change = abs(last_loglik_change) / (1.0 + abs(loglik))
        converged = (
            relative_loglik_change <= loglik_tol
            and last_item_change <= item_tol
            and last_latent_change <= latent_tol
        )

        if verbose and (iteration == 1 or iteration % 25 == 0 or converged):
            print(
                f"[MML] iter={iteration:4d} | logLik={loglik:.3f} | "
                f"dLL={last_loglik_change:.3g} | max|db|={last_item_change:.3g} | "
                f"latent change={last_latent_change:.3g}"
            )

        if converged:
            break

    # Final item-score diagnostics at the returned parameters.
    P = expit(grid[None, :] - b[:, None])
    raw_score = (n_jq * P).sum(axis=1) - r_jq.sum(axis=1)
    score = raw_score - raw_score.mean()  # mean(b)=0 constraint
    item_info = np.maximum((n_jq * P * (1.0 - P)).sum(axis=1), 1e-12)
    b_sd = 1.0 / np.sqrt(item_info)
    remaining_step = np.abs(score) / item_info
    remaining_step_in_se = remaining_step / b_sd

    if latent == "empirical":
        latent_target = np.maximum(post_sum / data.n_persons, 1e-300)
        latent_target /= latent_target.sum()
        final_latent_residual = float(0.5 * np.sum(np.abs(latent_target - weights)))
    else:
        check_mu, check_sigma, _, _ = _fit_discrete_normal(
            post_sum, grid
        )
        final_latent_residual = float(max(
            abs(check_mu - mu),
            abs(np.log(check_sigma) - np.log(sigma)),
        ))

    latent_mean = float(np.sum(weights * grid))
    latent_sd = float(np.sqrt(np.sum(weights * (grid - latent_mean) ** 2)))
    edge_mass = float(weights[:2].sum() + weights[-2:].sum())
    effective_nodes = float(1.0 / np.sum(weights ** 2))

    max_remaining_step = float(remaining_step.max())
    max_remaining_step_in_se = float(remaining_step_in_se.max())
    max_abs_score = float(np.max(np.abs(score)))

    items_params = {}
    for j, item_id in enumerate(item_list):
        items_params[item_id] = {
            "b": float(b[j]),
            "bSD": float(b_sd[j]),
            "bSD_approx?": True,
            "a": 1.0,
            "aSD": 0.0,
            "converged?": bool(converged),
            "correctResponses": int(n_correct.get(item_id, 0)),
            "totalResponses": int(n_responses.get(item_id, 0)),
            "scoreResidual": float(score[j]),
            "remainingStep": float(remaining_step[j]),
            "remainingStepInSE": float(remaining_step_in_se[j]),
        }

    info = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "n_items": data.J,
        "n_persons_used": data.n_persons,
        "n_responses_used": data.n_responses,
        "final_loglik": float(loglik),
        "last_loglik_change": last_loglik_change,
        "last_item_change": last_item_change,
        "last_latent_change": last_latent_change,
        "final_max_abs_score": max_abs_score,
        "final_max_step_logits": max_remaining_step,
        "final_max_step_in_SE": max_remaining_step_in_se,
        "final_latent_residual": final_latent_residual,
        "latent": latent,
        "latent_mean": latent_mean,
        "latent_sd": latent_sd,
        "latent_grid": grid.tolist(),
        "latent_weights": weights.tolist(),
        "latent_edge_mass": edge_mass,
        "latent_effective_nodes": effective_nodes,
        "normal_mstep_ok": bool(normal_mstep_ok),
    }

    if verbose:
        print(
            f"[MML] latent mean={latent_mean:.3f} SD={latent_sd:.3f} | "
            f"edge mass={edge_mass:.3g} | effective nodes={effective_nodes:.1f}"
        )
        print(
            f"[MML] largest remaining item step={max_remaining_step:.3g} logits "
            f"({max_remaining_step_in_se:.3g} item SE) | "
            f"latent residual={final_latent_residual:.3g}"
        )

    status = "MML converged" if converged else "MML did not converge"
    if edge_mass > 0.005:
        status += " | WARNING: latent mass near grid boundary; widen theta_range"
    if latent == "normal" and not normal_mstep_ok:
        status += " | WARNING: normal latent M-step optimizer reported failure"

    return items_params, bad_items, info, status


# ============================================================
# Person scoring with the fitted latent distribution
# ============================================================

def align_responses(persons, itemsParams, drop_empty=True):
    """Keep only responses to successfully calibrated items."""
    out = {}
    for person_id, responses in persons.items():
        retained = {
            item_id: int(value)
            for item_id, value in responses.items()
            if item_id in itemsParams
        }
        if retained or not drop_empty:
            out[person_id] = retained
    return out


def score_person_mml(
    person_responses,
    itemsParams,
    latent_grid,
    latent_weights,
    method="EAP",
):
    """Score one person using the fitted normal or empirical MML prior."""
    grid = np.asarray(latent_grid, dtype=float)
    weights = np.asarray(latent_weights, dtype=float)
    if grid.ndim != 1 or weights.shape != grid.shape:
        raise ValueError("latent_grid and latent_weights must be matching 1-D arrays")
    if not person_responses:
        raise ValueError("person_responses is empty")

    log_posterior = np.log(np.maximum(weights, 1e-300))
    for item_id, value in person_responses.items():
        b = float(itemsParams[item_id]["b"])
        z = grid - b
        if int(value) == 1:
            log_posterior += -np.logaddexp(0.0, -z)
        else:
            log_posterior += -np.logaddexp(0.0, z)

    log_posterior -= logsumexp(log_posterior)
    posterior = np.exp(log_posterior)

    method = method.upper()
    posterior_mean = float(np.sum(posterior * grid))
    if method == "EAP":
        theta = posterior_mean
    elif method == "MAP":
        theta = float(grid[int(np.argmax(posterior))])
    else:
        raise ValueError("method must be 'EAP' or 'MAP'")

    theta_sd = float(np.sqrt(np.sum(posterior * (grid - posterior_mean) ** 2)))
    return theta, theta_sd


def getItemPersonMMLParams(
    persons,
    items,
    model="1PL",
    min_responses=30,
    latent="normal",
    itemsInitialGuess=None,
    ability_estimation_method="EAP",
    grid=None,
    n_nodes=97,
    theta_range=(-14.0, 14.0),
    max_iter=1000,
    verbose=True,
):
    """Calibrate items, then score persons with the fitted latent distribution."""
    if model != "1PL":
        raise ValueError("This implementation supports only Rasch 1PL")

    itemsParams, badItems, info, item_status = calibrate_mml(
        persons,
        items,
        min_item_responses=min_responses,
        latent=latent,
        grid=grid,
        n_nodes=n_nodes,
        theta_range=theta_range,
        itemsInitialGuess=itemsInitialGuess,
        max_iter=max_iter,
        verbose=verbose,
    )

    scoreable = align_responses(persons, itemsParams, drop_empty=False)
    personsParams = {}
    n_extreme = 0
    n_empty = 0

    for person_id, responses in scoreable.items():
        if not responses:
            n_empty += 1
            continue
        total = len(responses)
        correct = sum(responses.values())
        theta, theta_sd = score_person_mml(
            responses,
            itemsParams,
            info["latent_grid"],
            info["latent_weights"],
            method=ability_estimation_method,
        )
        extreme = correct == 0 or correct == total
        n_extreme += int(extreme)
        personsParams[person_id] = {
            "theta": theta,
            "thetaSD": theta_sd,
            "extremeScore?": extreme,
            "converged?": True,
        }

    status = (
        f"{item_status} | {ability_estimation_method.upper()} person scores "
        f"(N={len(personsParams)}, extreme={n_extreme}, empty={n_empty}) | "
        f"fitted {latent} latent distribution"
    )
    return personsParams, itemsParams, badItems, status
