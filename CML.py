"""
============================================================
Rasch 1PL CML (Conditional Maximum Likelihood) item calibration
+ EAP person scoring (NO SD rescale)
============================================================

This is a Rasch CML implementation designed for CAT-sparse binary response data
stored in your nested-dict structures.

Data structures (your conventions):
    persons = {personID: {itemID: response_int_0_1}}
    items   = {itemID: {personID: response_int_0_1}}

Outputs (same style as your joint function):
    personsParams, itemsParams, badItems, status

Key properties:
- Item calibration uses Rasch CML (no priors, no person distribution assumption).
- Works with sparse CAT matrices naturally (each person only contributes over their
  administered items).
- Retired/removed items: if an itemID exists in person logs but not in `items`
  (your current bank), it is ignored deterministically.
- Identification: we center mean(b)=0 (NO SD rescale).
- Optimization: stable damped Newton with a diagonal Hessian approximation.
- Deterministic filtering:
    * items with < min_responses
    * items all-correct or all-incorrect in bank
    * persons with < 2 responses after filtering
    * persons with raw score r=0 or r=K (no info for CML)
    * disconnected components: keep largest connected component (optional but recommended)

You MUST provide in your codebase:
    getPersonAbility(personResponses, itemsParams, model, method='EAP',
                     thetaRange=(..), thetaSteps=.., priorMean=.., priorSigma=..)

============================================================
"""

from __future__ import annotations

import math
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Tuple, List

from irt import PCML
from irt.irt import getPersonAbility


# ============================================================
# Utilities
# ============================================================

def _logsumexp(vals: np.ndarray) -> float:
    m = float(np.max(vals))
    if not np.isfinite(m):
        return -np.inf
    return m + math.log(float(np.sum(np.exp(vals - m))))


def _logaddexp(a: float, b: float) -> float:
    # stable log(exp(a)+exp(b))
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a < b:
        a, b = b, a
    # a >= b
    return a + math.log1p(math.exp(b - a))


def _center_mean_zero(arr: np.ndarray) -> np.ndarray:
    return arr - float(arr.mean())


def _clip_b(b: np.ndarray, bmin: float, bmax: float) -> np.ndarray:
    return np.clip(b, bmin, bmax)


def _largest_connected_component_items(persons_filtered: Dict[Any, Dict[Any, int]], item_ids: List[Any]) -> List[Any]:
    adj = defaultdict(set)
    item_set = set(item_ids)

    for resp in persons_filtered.values():
        its = [it for it in resp.keys() if it in item_set]
        if len(its) < 2:
            continue
        for i in range(len(its)):
            a = its[i]
            for j in range(i + 1, len(its)):
                b = its[j]
                adj[a].add(b)
                adj[b].add(a)

    seen = set()
    best = []

    for start in item_ids:
        if start in seen:
            continue
        if start not in adj:
            seen.add(start)
            continue
        q = deque([start])
        comp = []
        seen.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        if len(comp) > len(best):
            best = comp

    return sorted(best)


# ============================================================
# Log-ESF recursion (stable)
# ============================================================

def _log_esf_forward_tables(logw: np.ndarray, rmax: int) -> List[np.ndarray]:
    """
    Forward recursion in log-space.
    F[t][k] = log ESF selecting k items among first t weights.
    """
    K = len(logw)
    F = [np.full(rmax + 1, -np.inf, dtype=float) for _ in range(K + 1)]
    F[0][0] = 0.0  # log(1)

    for t in range(1, K + 1):
        F[t][:] = F[t - 1]
        lw = float(logw[t - 1])
        for k in range(1, rmax + 1):
            # F[t][k] = logaddexp(F[t-1][k], lw + F[t-1][k-1])
            F[t][k] = _logaddexp(F[t][k], lw + float(F[t - 1][k - 1]))
    return F


def _log_esf_backward_tables(logw: np.ndarray, rmax: int) -> List[np.ndarray]:
    """
    Backward recursion in log-space.
    B[t][k] = log ESF selecting k items among weights t..K-1.
    """
    K = len(logw)
    B = [np.full(rmax + 1, -np.inf, dtype=float) for _ in range(K + 1)]
    B[K][0] = 0.0

    for t in range(K - 1, -1, -1):
        B[t][:] = B[t + 1]
        lw = float(logw[t])
        for k in range(1, rmax + 1):
            B[t][k] = _logaddexp(B[t][k], lw + float(B[t + 1][k - 1]))
    return B


def _log_gamma_excluding_index(F: List[np.ndarray], B: List[np.ndarray], idx_excl: int, r: int) -> float:
    """
    log gamma_r(excluding idx_excl) = log sum_{k=0..r} exp(F[idx_excl][k] + B[idx_excl+1][r-k])
    """
    if r < 0:
        return -np.inf
    vals = np.array([float(F[idx_excl][k]) + float(B[idx_excl + 1][r - k]) for k in range(r + 1)], dtype=float)
    return _logsumexp(vals)


# ============================================================
# CML calibration (log-stable)
# ============================================================

def cml_calibrate_items_rasch_1pl(
    persons: Dict[Any, Dict[Any, int]],
    items: Dict[Any, Dict[Any, int]],
    itemsInitialGuess: Dict[Any, Dict[str, float]] | None = None,
    min_item_responses: int = 2,
    min_person_responses: int = 2,
    keep_largest_component: bool = True,
    # optimization
    max_iter: int = 400,
    delta_b_max_tolerance: float = 1e-2,
    damping: float = 1,        # start smaller than 1.0
    max_step_halving: int = 40,
    hessian_floor: float = 1e-5,
    step_cap: float = 5.0,        # cap per-iteration max |Δb|
    b_clip: Tuple[float, float] = (-15.0, 15.0),  # prevent runaway early
    verbose: bool = True,
) -> Tuple[Dict[Any, Dict[str, float]], Dict[Any, Dict[str, Any]], Dict[str, Any], str]:
    """
    Rasch CML with stable log-ESF computations.
    """

    # ---- item filtering and initial guesses ----
    itemsInitialGuess = itemsInitialGuess or {}
    itemsParams: Dict[Any, Dict[str, float]] = {}
    badItems: Dict[Any, Dict[str, Any]] = {}
    
    for itemID, resp in items.items():
        n = len(resp)
        if n < min_item_responses:
            badItems[itemID] = {"reason": "too_few_responses", "totalResponses": n}
            continue
        c = sum(resp.values())
        if c == 0:
            badItems[itemID] = {"reason": "all_incorrect", "totalResponses": n}
            continue
        if c == n:
            badItems[itemID] = {"reason": "all_correct", "totalResponses": n}
            continue
        b0 = math.log((n - c) / c)

        # check if initial value of b is provided
        if itemID in itemsInitialGuess:
            if 'b' in itemsInitialGuess[itemID]:
                b_initial_guess = itemsInitialGuess[itemID]['b']
                if isinstance(b_initial_guess, (int, float, np.number)) and np.isfinite(b_initial_guess):
                    b0 = b_initial_guess

        itemsParams[itemID] = {
            "b": float(b0), 
            "bSD": float("nan"),
            "a": 1.0,
            "aSD": 0.0,
            "converged?": False,
            "correctResponses": int(c),
            "totalResponses": int(n),
        }

    if len(itemsParams) < 3:
        raise ValueError("CML: too few usable items after basic filtering.")

    usable_items = set(itemsParams.keys())

    # ---- person filtering (drop retired items deterministically) ----
    persons_f: Dict[Any, Dict[Any, int]] = {}
    dropped_retired = 0
    dropped_person_short = 0

    for pid, resp in persons.items():
        resp2 = {it: x for it, x in resp.items() if it in usable_items}
        dropped_retired += (len(resp) - len(resp2))
        if len(resp2) < min_person_responses:
            dropped_person_short += 1
            continue
        persons_f[pid] = resp2

    if len(persons_f) < 10:
        raise ValueError("CML: too few persons after filtering.")

    # ---- keep largest connected component ----
    item_list = sorted(usable_items)
    if keep_largest_component:
        comp = _largest_connected_component_items(persons_f, item_list)
        comp_set = set(comp)
        dropped = [it for it in item_list if it not in comp_set]
        for it in dropped:
            badItems[it] = {"reason": "disconnected_component"}
            itemsParams.pop(it, None)
        usable_items = set(itemsParams.keys())
        item_list = sorted(usable_items)

        # filter persons again
        persons_f2 = {}
        for pid, resp in persons_f.items():
            rr = {it: x for it, x in resp.items() if it in usable_items}
            if len(rr) >= min_person_responses:
                persons_f2[pid] = rr
        persons_f = persons_f2

    if len(item_list) < 3:
        raise ValueError("CML: too few items after component filtering.")

    # ---- index mapping ----
    idx = {it: i for i, it in enumerate(item_list)}
    J = len(item_list)

    # init b
    b = np.array([float(itemsParams[it]["b"]) for it in item_list], dtype=float)
    b = _center_mean_zero(_clip_b(b, b_clip[0], b_clip[1]))

    # ---- pack persons ----
    packed = []
    dropped_all_same = 0
    for pid, resp in persons_f.items():
        its, xs = [], []
        for it, x in resp.items():
            if it in idx:
                its.append(idx[it])
                xs.append(int(x))
        K = len(xs)
        if K < min_person_responses:
            continue
        r = int(sum(xs))
        if r == 0 or r == K:
            dropped_all_same += 1
            continue
        packed.append((pid, np.array(its, dtype=int), np.array(xs, dtype=int), r, K))

    if len(packed) < 10:
        raise ValueError("CML: too few informative persons (many are all-0/all-1).")

    # ---- objective ----
    def eval_ll_grad_hdiag(b_vec: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        ll = 0.0
        grad = np.zeros(J, dtype=float)
        hdiag = np.zeros(J, dtype=float)

        for _, its, xs, r, K in packed:
            b_sub = b_vec[its]
            # logw = eta = -b
            logw = -_clip_b(b_sub, b_clip[0], b_clip[1])

            # ESF in log-space up to r
            F = _log_esf_forward_tables(logw, rmax=r)
            B = _log_esf_backward_tables(logw, rmax=r)

            log_gamma_r = float(F[K][r])
            if not np.isfinite(log_gamma_r):
                continue

            # loglik for person: sum_correct eta - log_gamma_r
            ll += float(np.dot(xs, logw) - log_gamma_r)

            # expected inclusion probs
            for t in range(K):
                it_global = its[t]
                if r - 1 < 0:
                    p = 0.0
                else:
                    log_gamma_excl = _log_gamma_excluding_index(F, B, idx_excl=t, r=r - 1)
                    # p = exp(logw[t] + log_gamma_excl - log_gamma_r)
                    logp = float(logw[t]) + float(log_gamma_excl) - float(log_gamma_r)
                    # clamp logp to avoid numerical junk
                    if logp < -50:
                        p = 0.0
                    elif logp > 0:
                        # p cannot exceed 1; if >0 it's >1, clamp
                        p = 1.0
                    else:
                        p = float(math.exp(logp))

                # grad wrt b: (p - x)
                grad[it_global] += (p - xs[t])
                hdiag[it_global] += max(p * (1.0 - p), 0.0)

        hdiag = np.maximum(hdiag, hessian_floor)
        return ll, grad, hdiag

    ll, grad, hdiag = eval_ll_grad_hdiag(b)
    if verbose:
        print(f"[CML] items kept: {J} | persons used: {len(packed)}")
        print(f"[CML] dropped retired responses: {dropped_retired:,} | dropped persons short: {dropped_person_short:,} | dropped all-0/all-1: {dropped_all_same:,}")
        print(f"[CML] iter=000 ll={ll:.6f} max|grad|={float(np.max(np.abs(grad))):.6g}")

    converged = False
    itn_done = 0

    for itn in range(1, max_iter + 1):
        itn_done = itn

        step = +grad / hdiag

        # cap step magnitude deterministically
        m = float(np.max(np.abs(step)))
        if m > step_cap:
            step *= (step_cap / m)

        alpha = damping
        accepted = False
        b_new = b.copy()

        for _ in range(max_step_halving + 1):
            b_try = b + alpha * step
            b_try = _center_mean_zero(_clip_b(b_try, b_clip[0], b_clip[1]))

            ll_try, grad_try, hdiag_try = eval_ll_grad_hdiag(b_try)

            # accept if improved (or not worse within tiny epsilon)
            if np.isfinite(ll_try) and ll_try >= ll - 1e-8:
                b_new = b_try
                ll_new, grad_new, hdiag_new = ll_try, grad_try, hdiag_try
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            if verbose:
                print(f"[CML] iter={itn:03d} step rejected repeatedly; stopping.")
            break

        b = b_new
        ll, grad, hdiag = ll_new, grad_new, hdiag_new

        max_db = float(np.max(np.abs(alpha * step)))
        max_grad = float(np.max(np.abs(grad)))

        if verbose and (itn == 1 or itn % 10 == 0 or max_db < delta_b_max_tolerance):
            print(f"[CML] iter={itn:03d} ll={ll:.6f} max|Δb|={max_db:.6g} max|grad|={max_grad:.6g} alpha={alpha:.3g}")

        if max_db < delta_b_max_tolerance:
            converged = True
            break

    # final SEs
    ll, grad, hdiag = eval_ll_grad_hdiag(b)
    se = np.sqrt(1.0 / np.maximum(hdiag, hessian_floor))

    for it in item_list:
        i = idx[it]
        itemsParams[it]["b"] = float(b[i])
        itemsParams[it]["bSD"] = float(se[i])
        itemsParams[it]["a"] = 1.0
        itemsParams[it]["aSD"] = 0.0
        itemsParams[it]["converged?"] = bool(converged)

    info = {
        "iterations": int(itn_done),
        "converged": bool(converged),
        "n_items": int(J),
        "n_persons_used": int(len(packed)),
        "dropped_retired_responses": int(dropped_retired),
        "dropped_persons_short": int(dropped_person_short),
        "dropped_persons_all0_all1": int(dropped_all_same),
        "final_loglik": float(ll),
        "final_max_abs_grad": float(np.max(np.abs(grad))) if len(grad) else float("nan"),
    }
    status = "CML converged" if converged else "CML stopped (not fully converged)"
    return itemsParams, badItems, info, status


# ============================================================
# Full pipeline: CML items + EAP persons + mean centering (NO SD rescale)
# ============================================================

def getItemPersonCMLParams(
    persons: Dict[Any, Dict[Any, int]],
    items: Dict[Any, Dict[Any, int]],
    model: str = "1PL",
    itemsInitialGuess: Dict[Any, Dict[str, float]] | None = None,
    # CML filtering + optimization
    min_item_responses: int = 2,
    min_person_responses: int = 2,
    cml_max_iter: int = 1500,
    delta_b_max_tolerance: float = 1e-3,
    keep_largest_component: bool = True,
    verbose: bool = True,
    # Person scoring (EAP) settings
    ranges: Dict[str, Tuple[float, float]] = {"theta": (-12, 12)},
    steps: Dict[str, int] = {"theta": 25},
    priorMean: Dict[str, float] = {"theta": 0.0},
    priorSigma: Dict[str, float] = {"theta": 6.0},
):
    if model != "1PL":
        raise ValueError("This CML pipeline supports only Rasch 1PL.")
    
    ability_estimation_method = 'MLE'

    itemsParams, badItems, info, status_items = cml_calibrate_items_rasch_1pl(
        persons=persons,
        items=items,
        itemsInitialGuess=itemsInitialGuess,
        min_item_responses=min_item_responses,
        min_person_responses=min_person_responses,
        keep_largest_component=keep_largest_component,
        max_iter=cml_max_iter,
        delta_b_max_tolerance=delta_b_max_tolerance,
        verbose=verbose,
    )

    personsParams: Dict[Any, Dict[str, float]] = {}
    theta_range = ranges.get("theta", (-12, 12))
    theta_steps = int(steps.get("theta", 25))
    theta_prior_mean = float(priorMean.get("theta", 0.0))
    theta_prior_sigma = float(priorSigma.get("theta", 6.0))

    for pid, resp in persons.items():
        resp_f = {it: x for it, x in resp.items() if it in itemsParams}
        if len(resp_f) < 1:
            continue

        theta, thetaSD = getPersonAbility(
            resp_f,
            itemsParams,
            model="1PL",
            method=ability_estimation_method,
            thetaRange=theta_range,
            thetaSteps=theta_steps,
            priorMean=theta_prior_mean,
            priorSigma=theta_prior_sigma,
        )

        personsParams[pid] = {
            "theta": float(theta),
            "thetaSD": float(thetaSD),
            "converged?": True,
        }

    # Center mean(theta)=0 (no SD rescale)
    if personsParams:
        theta_vals = np.array([personsParams[p]["theta"] for p in personsParams], dtype=float)
        theta_mean = float(theta_vals.mean())
    else:
        theta_mean = 0.0

    if verbose:
        print(f"[CML] Centering mean(theta)={theta_mean:.6g} -> 0 (SD unchanged)")

    for pid in personsParams:
        personsParams[pid]["theta"] -= theta_mean
    for it in itemsParams:
        itemsParams[it]["b"] -= theta_mean

    status = f"{status_items} | {ability_estimation_method} persons done (N={len(personsParams)}) | centered mean(theta)=0"
    return personsParams, itemsParams, badItems, status
