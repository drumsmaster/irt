"""
============================================================
PCML Rasch (1PL) calibration + EAP person scoring (NO SD rescale)
============================================================

Stable, CAT-friendly PCML using your nested dict structures, with deterministic
handling of "retired" / removed items that may still exist in person logs.

Data structures:
    persons = {personID: {itemID: response_int_0_1}}
    items   = {itemID: {personID: response_int_0_1}}

Key behaviors:
- Items that are present in persons logs but NOT present in current bank (items dict)
  are deterministically ignored everywhere (PCML + EAP scoring).
- Item calibration uses WEIGHTED PCML-MM (Bradley–Terry MM), so each informative
  person contributes total weight ~= 1. This reduces CAT-length overweighting
  and typically brings the scale closer to TAM.
- Items are filtered deterministically:
  * too few responses (min_responses)
  * all-correct / all-incorrect (in the bank data)
  * no pairwise information after filtering
  * separation (wins==0 or losses==0 in pairwise data)
  * near-separation (extreme winrate)
  * disconnected components (keep largest connected component only)
- Identification: mean(b)=0 inside PCML iterations; final pipeline centers mean(theta)=0
  by shifting theta and b by the same constant. NO SD rescaling.

You must provide in your codebase:
- getPersonAbility(personResponses, itemsParams, model, method='EAP',
                   thetaRange=(..), thetaSteps=.., priorMean=.., priorSigma=..)

============================================================
"""

from __future__ import annotations

import math
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Tuple, Any, List
from irt.irt import getPersonAbility


# ============================================================
# Utilities
# ============================================================

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def filter_pair_dict_by_item_set(
    pair_dict: Dict[Tuple[Any, Any], float],
    item_set: set,
) -> Dict[Tuple[Any, Any], float]:
    """Keep only pairs (a,b) where both endpoints are in item_set."""
    out = {}
    for (a, b), v in pair_dict.items():
        if a in item_set and b in item_set:
            out[(a, b)] = v
    return out


def filter_wins_by_item_set(
    wins: Dict[Tuple[Any, Any], float],
    item_set: set,
) -> Dict[Tuple[Any, Any], float]:
    """Keep only directed wins (j,k) where both endpoints are in item_set."""
    out = {}
    for (j, k), v in wins.items():
        if j in item_set and k in item_set and j != k:
            out[(j, k)] = v
    return out


# ============================================================
# Pairwise data construction (WEIGHTED, ignores retired items)
# ============================================================

def pcml_collect_pair_counts_weighted(
    persons: Dict[Any, Dict[Any, int]],
    active_items: set,
    per_person_total_weight: float = 1.0
) -> Tuple[Dict[Tuple[Any, Any], float], set, float, Dict[str, int]]:
    """
    Weighted pair construction:
      For each person i:
        weight_per_pair = per_person_total_weight / (|C_i| * |W_i|)
      wins[(j,k)] += weight_per_pair for each correct j and incorrect k.

    Only items in active_items are considered (retired items ignored).
    """
    wins = defaultdict(float)
    items_seen = set()
    total_weight = 0.0

    dropped_responses_not_in_bank = 0
    persons_with_no_pairs_after_filtering = 0

    for _, resp in persons.items():
        resp_f = {it: x for it, x in resp.items() if it in active_items}
        dropped_responses_not_in_bank += (len(resp) - len(resp_f))

        correct = [it for it, x in resp_f.items() if x == 1]
        incorrect = [it for it, x in resp_f.items() if x == 0]
        if not correct or not incorrect:
            persons_with_no_pairs_after_filtering += 1
            continue

        c = len(correct)
        w = len(incorrect)
        weight = per_person_total_weight / (c * w)

        for j in correct:
            items_seen.add(j)
            for k in incorrect:
                items_seen.add(k)
                wins[(j, k)] += weight
                total_weight += weight

    diag = {
        "dropped_responses_not_in_bank": dropped_responses_not_in_bank,
        "persons_with_no_pairs_after_filtering": persons_with_no_pairs_after_filtering,
    }
    return dict(wins), items_seen, float(total_weight), diag


def _build_undirected_counts_weighted(
    wins: Dict[Tuple[Any, Any], float],
    item_set: set
) -> Tuple[Dict[Tuple[Any, Any], float], Dict[Tuple[Any, Any], float], Dict[Any, float], Dict[Any, float]]:
    """
    Build undirected pair totals and oriented win totals for i<j, using float weights.

    Outputs:
      n[(i,j)] = wins(i,j) + wins(j,i)    for i<j
      w[(i,j)] = wins(i,j)                for i<j
      wins_total[item]   = total weighted wins
      losses_total[item] = total weighted losses
    """
    n = defaultdict(float)
    w = defaultdict(float)
    wins_total = defaultdict(float)
    losses_total = defaultdict(float)

    for (j, k), c in wins.items():
        if j not in item_set or k not in item_set or j == k:
            continue
        a, b = (j, k) if j < k else (k, j)
        n[(a, b)] += c
        if j < k:
            w[(a, b)] += c
        wins_total[j] += c
        losses_total[k] += c

    for it in item_set:
        wins_total[it] += 0.0
        losses_total[it] += 0.0

    return dict(n), dict(w), dict(wins_total), dict(losses_total)


def _largest_connected_component(
    nodes: List[Any],
    n_pairs: Dict[Tuple[Any, Any], float],
    allowed_set: set,
) -> List[Any]:
    """
    Largest connected component over the undirected graph induced by allowed_set.
    We DO NOT traverse to neighbors outside allowed_set.
    """
    adj = defaultdict(list)
    for (a, b), cnt in n_pairs.items():
        if cnt <= 0:
            continue
        if a in allowed_set and b in allowed_set:
            adj[a].append(b)
            adj[b].append(a)

    seen = set()
    best = []

    for start in nodes:
        if start not in allowed_set:
            continue
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
                if v in allowed_set and v not in seen:
                    seen.add(v)
                    q.append(v)

        if len(comp) > len(best):
            best = comp

    return sorted(best)


# ============================================================
# PCML item calibration (stable MM solver, weighted)
# ============================================================

def pcml_estimate_item_b_mm_weighted(
    persons: Dict[Any, Dict[Any, int]],
    items: Dict[Any, Dict[Any, int]],
    itemsInitialGuess: Dict[Any, Dict[str, float]] | None = None,
    min_responses: int = 2,
    per_person_total_weight: float = 1.0,
    max_iter: int = 3000,
    tol: float = 1e-6,
    eps: float = 1e-12,
    min_pair_mass_per_item: float = 5.0,
    near_sep_winrate_eps: float = 1e-4,
    verbose: bool = True,
):
    """
    Weighted PCML via Bradley–Terry MM.

    IMPORTANT:
      - Retired items (present in persons logs but not in items dict) are ignored.
      - Graph/edge dicts are filtered after each pruning step to avoid the exact
        crash you are seeing.
    """
    itemsInitialGuess = itemsInitialGuess or {}

    # ---- 1) Build itemsParams from ACTIVE BANK (items dict), with deterministic filtering ----
    itemsParams: Dict[Any, Dict[str, float]] = {}
    badItems: Dict[Any, Dict[str, Any]] = {}

    for itemID, resp in items.items():
        total = len(resp)
        if total < min_responses:
            badItems[itemID] = {"reason": "too_few_responses", "totalResponses": total}
            continue
        correct = sum(resp.values())
        if correct == 0:
            badItems[itemID] = {"reason": "all_incorrect", "totalResponses": total}
            continue
        if correct == total:
            badItems[itemID] = {"reason": "all_correct", "totalResponses": total}
            continue

        b0 = math.log((total - correct) / correct)
        if itemID in itemsInitialGuess:
            b0 = float(itemsInitialGuess[itemID].get("b", b0))

        itemsParams[itemID] = {
            "b": float(b0),
            "bSD": float("nan"),
            "a": 1.0,
            "aSD": 0.0,
            "converged?": False,
            "correctResponses": int(correct),
            "totalResponses": int(total),
        }

    if len(itemsParams) < 3:
        raise ValueError("PCML-MM weighted: too few usable items after basic filtering.")

    # ---- 2) Pairwise wins from persons, restricted to active+usable items ----
    active_for_pairs = set(itemsParams.keys())
    wins, items_seen, total_pair_weight, diag_pairs = pcml_collect_pair_counts_weighted(
        persons=persons,
        active_items=active_for_pairs,
        per_person_total_weight=per_person_total_weight
    )

    if verbose:
        print(f"[PCML] dropped responses not in bank: {diag_pairs['dropped_responses_not_in_bank']:,}")
        print(f"[PCML] persons with no pairs after filtering: {diag_pairs['persons_with_no_pairs_after_filtering']:,}")

    # Keep only items that appear in at least one comparison
    item_set = set(itemsParams.keys()) & set(items_seen)

    for it in list(itemsParams.keys()):
        if it not in item_set:
            badItems[it] = {"reason": "no_pairwise_information"}
            itemsParams.pop(it, None)

    if len(item_set) < 3:
        raise ValueError("PCML-MM weighted: too few items with pairwise information.")

    # Filter wins to current item_set (important for later pruning)
    wins = filter_wins_by_item_set(wins, item_set)

    # ---- 3) Build undirected totals from CURRENT item_set ----
    n_pairs, w_pairs, wins_total, losses_total = _build_undirected_counts_weighted(wins, item_set)

    # ---- 3a) Drop weak pair mass items ----
    tot_mass = {it: wins_total[it] + losses_total[it] for it in item_set}
    weak = [it for it in item_set if tot_mass[it] < min_pair_mass_per_item]
    for it in weak:
        badItems[it] = {"reason": "too_little_pairwise_mass", "pair_mass": float(tot_mass[it])}
        itemsParams.pop(it, None)
        item_set.discard(it)

    if len(item_set) < 3:
        raise ValueError("PCML-MM weighted: too few items after pair-mass filtering.")

    # IMPORTANT: re-filter wins and rebuild pair structures after pruning
    wins = filter_wins_by_item_set(wins, item_set)
    n_pairs, w_pairs, wins_total, losses_total = _build_undirected_counts_weighted(wins, item_set)

    # ---- 3b) Separation (wins==0 or losses==0) ----
    sep = [it for it in item_set if (wins_total[it] == 0.0 or losses_total[it] == 0.0)]
    for it in sep:
        badItems[it] = {"reason": "pairwise_separation", "wins": float(wins_total[it]), "losses": float(losses_total[it])}
        itemsParams.pop(it, None)
        item_set.discard(it)

    if len(item_set) < 3:
        raise ValueError("PCML-MM weighted: too few items after separation filtering.")

    wins = filter_wins_by_item_set(wins, item_set)
    n_pairs, w_pairs, wins_total, losses_total = _build_undirected_counts_weighted(wins, item_set)

    # ---- 3c) Near separation (extreme winrate) ----
    near = []
    for it in item_set:
        t = wins_total[it] + losses_total[it]
        if t <= 0:
            continue
        wr = wins_total[it] / t
        if wr < near_sep_winrate_eps or wr > (1.0 - near_sep_winrate_eps):
            near.append((it, wr, t))

    for it, wr, t in near:
        badItems[it] = {"reason": "near_separation", "winrate": float(wr), "pair_mass": float(t)}
        itemsParams.pop(it, None)
        item_set.discard(it)

    if len(item_set) < 3:
        raise ValueError("PCML-MM weighted: too few items after near-separation filtering.")

    wins = filter_wins_by_item_set(wins, item_set)
    n_pairs, w_pairs, wins_total, losses_total = _build_undirected_counts_weighted(wins, item_set)

    # ---- 4) Connectivity: keep largest connected component induced by item_set ----
    n_pairs = filter_pair_dict_by_item_set(n_pairs, item_set)

    item_list = sorted(item_set)
    comp = _largest_connected_component(item_list, n_pairs, allowed_set=item_set)
    dropped = sorted(set(item_list) - set(comp))
    for it in dropped:
        badItems[it] = {"reason": "disconnected_component"}
        itemsParams.pop(it, None)
        item_set.discard(it)

    if len(item_set) < 3:
        raise ValueError("PCML-MM weighted: largest connected component too small.")

    # Finalize to component only (and filter pairs again)
    item_list = sorted(item_set)
    n_pairs = filter_pair_dict_by_item_set(n_pairs, item_set)

    # Bulletproof: ensure item_list all exist in itemsParams
    item_list = [it for it in item_list if it in itemsParams]
    missing_now = sorted(list(item_set - set(item_list)))
    for it in missing_now:
        badItems[it] = {"reason": "inconsistent_state_missing_itemsParams"}
        item_set.discard(it)

    J = len(item_list)
    if J < 3:
        raise ValueError("PCML-MM weighted: too few items after consistency filtering.")

    idx = {it: i for i, it in enumerate(item_list)}

    # ---- 5) Initialize lambdas = exp(-b) ----
    b0 = np.array([float(itemsParams[it]["b"]) for it in item_list], dtype=float)
    b0 -= b0.mean()
    lam = np.exp(-np.clip(b0, -50, 50))

    # Build neighbor lists from n_pairs (safe: all endpoints in item_list)
    neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(J)]
    for (a, b), nij in n_pairs.items():
        if nij <= 0:
            continue
        if a in idx and b in idx:
            ia, ib = idx[a], idx[b]
            neighbors[ia].append((ib, float(nij)))
            neighbors[ib].append((ia, float(nij)))

    w_item = np.array([float(wins_total[it]) for it in item_list], dtype=float)

    if verbose:
        n_unique_pairs = sum(1 for c in n_pairs.values() if c > 0)
        print(f"[PCML-MM weighted] items kept: {J} | badItems: {len(badItems)}")
        print(f"[PCML-MM weighted] unique pairs: {n_unique_pairs:,} | total pair-weight: {total_pair_weight:,.2f}")

    # ---- 6) MM iterations ----
    itn = 0
    for itn in range(1, max_iter + 1):
        lam_old = lam.copy()

        for i in range(J):
            denom = 0.0
            li = lam[i]
            for j, nij in neighbors[i]:
                denom += nij / (li + lam[j])
            denom = max(denom, eps)
            lam[i] = max(w_item[i] / denom, eps)

        # identification: mean(b)=0 <=> mean(log lam)=0
        loglam = np.log(lam)
        loglam -= loglam.mean()
        lam = np.exp(loglam)

        max_rel = float(np.max(np.abs(lam - lam_old) / (np.abs(lam_old) + 1e-12)))
        if verbose and (itn == 1 or itn % 100 == 0 or max_rel < tol):
            print(f"[PCML-MM weighted] iter={itn:04d} max_rel_change(lambda)={max_rel:.6g}")
        if max_rel < tol:
            break

    b = -np.log(lam)
    b -= b.mean()

    # ---- 7) Approx SEs ----
    info_diag = np.zeros(J, dtype=float)
    for i in range(J):
        bi = b[i]
        for j, nij in neighbors[i]:
            bj = b[j]
            p = _sigmoid(bj - bi)  # P(i beats j)
            info_diag[i] += nij * p * (1.0 - p)
    se = np.sqrt(1.0 / np.maximum(info_diag, 1e-9))

    # Write back
    for it in item_list:
        i = idx[it]
        itemsParams[it]["b"] = float(b[i])
        itemsParams[it]["bSD"] = float(se[i])
        itemsParams[it]["a"] = 1.0
        itemsParams[it]["aSD"] = 0.0
        itemsParams[it]["converged?"] = True

    info = {
        "iterations": int(itn),
        "n_items": int(J),
        "total_pair_weight": float(total_pair_weight),
        "dropped_responses_not_in_bank": int(diag_pairs["dropped_responses_not_in_bank"]),
        "persons_with_no_pairs_after_filtering": int(diag_pairs["persons_with_no_pairs_after_filtering"]),
    }
    status = f"PCML-MM weighted converged (iter={itn}, items={J})"
    return itemsParams, badItems, info, status


# ============================================================
# Full pipeline: weighted PCML items + EAP persons + mean centering
# ============================================================

def getItemPersonPCMLParams(
    persons: Dict[Any, Dict[Any, int]],
    items: Dict[Any, Dict[Any, int]],
    model: str = "1PL",
    ranges: Dict[str, Tuple[float, float]] = {"theta": (-12, 12)},
    steps: Dict[str, int] = {"theta": 25},
    priorMean: Dict[str, float] = {"theta": 0.0},
    priorSigma: Dict[str, float] = {"theta": 6.0},
    itemsInitialGuess: Dict[Any, Dict[str, float]] | None = None,
    personsInitialGuess: Dict[Any, Dict[str, float]] | None = None,
    min_responses: int = 2,
    per_person_total_weight: float = 1.0,
    min_pair_mass_per_item: float = 5.0,
    near_sep_winrate_eps: float = 1e-4,
    pcml_max_iter: int = 3000,
    pcml_tol: float = 1e-6,
    pcml_eps: float = 1e-12,
    verbose: bool = True,
):
    if model != "1PL":
        raise ValueError("PCML pipeline supports only Rasch 1PL.")

    itemsInitialGuess = itemsInitialGuess or {}
    personsInitialGuess = personsInitialGuess or {}

    # ---- 1) Items: weighted PCML ----
    itemsParams, badItems, info, status_items = pcml_estimate_item_b_mm_weighted(
        persons=persons,
        items=items,
        itemsInitialGuess=itemsInitialGuess,
        min_responses=min_responses,
        per_person_total_weight=per_person_total_weight,
        max_iter=pcml_max_iter,
        tol=pcml_tol,
        eps=pcml_eps,
        min_pair_mass_per_item=min_pair_mass_per_item,
        near_sep_winrate_eps=near_sep_winrate_eps,
        verbose=verbose,
    )

    # ---- 2) Persons: EAP scoring with fixed items ----
    personsParams: Dict[Any, Dict[str, float]] = {}

    theta_range = ranges.get("theta", (-12, 12))
    theta_steps = int(steps.get("theta", 25))
    theta_prior_mean = float(priorMean.get("theta", 0.0))
    theta_prior_sigma = float(priorSigma.get("theta", 6.0))

    for personID, resp in persons.items():
        resp_f = {it: x for it, x in resp.items() if it in itemsParams}
        if len(resp_f) == 0:
            continue

        _ = personsInitialGuess.get(personID, {}).get("theta", 0.0)

        theta, thetaSD = getPersonAbility(
            resp_f,
            itemsParams,
            model="1PL",
            method="EAP",
            thetaRange=theta_range,
            thetaSteps=theta_steps,
            priorMean=theta_prior_mean,
            priorSigma=theta_prior_sigma,
        )

        personsParams[personID] = {
            "theta": float(theta),
            "thetaSD": float(thetaSD),
            "converged?": True,
        }

    # ---- 3) Center mean(theta)=0 ONLY ----
    if personsParams:
        theta_vals = np.array([personsParams[p]["theta"] for p in personsParams], dtype=float)
        theta_mean = float(theta_vals.mean())
    else:
        theta_mean = 0.0

    if verbose:
        print(f"[PCML] Centering mean(theta)={theta_mean:.6g} -> 0 (SD unchanged)")

    for pid in personsParams:
        personsParams[pid]["theta"] -= theta_mean
    for it in itemsParams:
        itemsParams[it]["b"] -= theta_mean

    status = status_items + f" | EAP persons done (N={len(personsParams)}) | centered mean(theta)=0"
    return personsParams, itemsParams, badItems, status
