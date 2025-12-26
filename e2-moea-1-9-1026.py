# -*- coding: utf-8 -*-
import os, sys, math, argparse, json, time, random, csv, glob
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ Shared utils ============================
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)

def ensure_dir(p:str):
    os.makedirs(p, exist_ok=True)

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def save_csv(path: str, arr: np.ndarray, header: Optional[str] = None):
    if header is None:
        np.savetxt(path, arr, delimiter=",")
    else:
        np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.10g")

def lhs(n, d, rng):
    """Latin Hypercube Sampling in [0,1]^d."""
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        rng.shuffle(u[:, j])
    return u

def opposition_around(mu, x, lb, ub):
    """Opposition sampling around mean mu (mirror w.r.t mu, then clamp)."""
    return np.clip(2.0 * mu - x, lb, ub)

# ============================ Official DAS-CMOP1..9 ============================
def build_dascmop(problem="dascmop1", n_var=30, triplet=(0.5, 0.5, 0.5)):
    """
    Returns (evaluate, lb, ub, n_obj)
    evaluate(X) -> (F, G)  (G>=0 feasible)
    """
    problem = problem.lower()
    eta, zeta, gamma = triplet

    # constants
    a_freq = 20.0
    d = 0.5
    theta = -0.25 * math.pi
    ak = math.sqrt(0.3)      # ellipse radii
    bk = math.sqrt(1.2)

    # triplet -> (b, e, r)
    b = 2.0 * eta - 1.0
    e = (1e9 if zeta <= 0 else d - math.log(zeta))
    r = 0.5 * gamma

    # centers of 9 rotated ellipses (2D cases)
    pk = np.array([0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=float)
    qk = np.array([1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5], dtype=float)

    # g(x) terms
    def g_1to3(x):
        x = np.atleast_2d(x); x1 = x[:, [0]]
        return np.sum((x - np.sin(0.5 * math.pi * x1)) ** 2, axis=1)

    def g_4to6(x):
        x = np.atleast_2d(x)
        if x.shape[1] < 2: return np.zeros(x.shape[0])
        y = x[:, 1:] - 0.5
        return (x.shape[1] - 1) + np.sum(y ** 2 - np.cos(20 * math.pi * y), axis=1)

    def g_7to8(x):
        x = np.atleast_2d(x)
        if x.shape[1] < 3: return np.zeros(x.shape[0])
        y = x[:, 2:] - 0.5
        return (x.shape[1] - 2) + np.sum(y ** 2 - np.cos(20 * math.pi * y), axis=1)

    def g_9(x):
        x = np.atleast_2d(x); n = x.shape[1]
        if n < 3: return np.zeros(x.shape[0])
        idx = np.arange(3, n + 1)
        x1 = x[:, 0:1]; x2 = x[:, 1:2]
        phase = 0.25 * (idx / n) * math.pi * (x1 + x2)
        target = np.cos(phase)
        y = x[:, 2:] - target
        return np.sum(y ** 2, axis=1)

    # constraints
    def bi_constraints(F, gvals, x):
        x = np.atleast_2d(x); N = x.shape[0]
        G = np.zeros((N, 11))
        x1 = x[:, 0]
        G[:, 0] = np.sin(a_freq * math.pi * x1) - b
        G[:, 1] = (e - gvals) * (gvals - d)
        f1, f2 = F[:, 0], F[:, 1]
        c, s = math.cos(theta), math.sin(theta)
        for k in range(9):
            u = f1 - pk[k]; v = f2 - qk[k]
            Xc = u * c + v * s
            Yc = -u * s + v * c
            ell = (Xc ** 2) / (ak ** 2) + (Yc ** 2) / (bk ** 2) - r
            G[:, 2 + k] = ell
        return G

    def tri_constraints(F, gvals, x):
        x = np.atleast_2d(x); N = x.shape[0]
        G = np.zeros((N, 7))
        x1, x2 = x[:, 0], x[:, 1]
        G[:, 0] = np.sin(a_freq * math.pi * x1) - b
        G[:, 1] = np.cos(a_freq * math.pi * x2) - b
        G[:, 2] = (e - gvals) * (gvals - d)
        # spheres centered at e1/e2/e3 (radius r) in F-space
        for k in range(3):
            fk = F[:, k]
            s2 = np.sum(F ** 2, axis=1) - fk ** 2 + (fk - 1.0) ** 2
            G[:, 3 + k] = s2 - (r ** 2)
        # sphere centered at (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        center = 1.0 / math.sqrt(3.0)
        G[:, 6] = np.sum((F - center) ** 2, axis=1) - (r ** 2)
        return G

    def evaluate(X):
        X = np.clip(np.atleast_2d(X), 0.0, 1.0)
        x1 = X[:, 0]; prob = problem

        if prob in ("dascmop1", "das-cmop1", "cmop1"):
            g = g_1to3(X); f1 = x1 + g; f2 = 1.0 - x1 ** 2 + g
            F = np.vstack([f1, f2]).T; G = bi_constraints(F, g, X)

        elif prob in ("dascmop2", "das-cmop2", "cmop2"):
            g = g_1to3(X); f1 = x1 + g; f2 = 1.0 - np.sqrt(np.maximum(x1, 0.0)) + g
            F = np.vstack([f1, f2]).T; G = bi_constraints(F, g, X)

        elif prob in ("dascmop3", "das-cmop3", "cmop3"):
            g = g_1to3(X); f1 = x1 + g
            f2 = 1.0 - np.sqrt(np.maximum(x1, 0.0)) + 0.5 * np.abs(np.sin(5 * math.pi * x1)) + g
            F = np.vstack([f1, f2]).T; G = bi_constraints(F, g, X)

        elif prob in ("dascmop4", "das-cmop4", "cmop4"):
            g = g_4to6(X); f1 = x1 + g; f2 = 1.0 - x1 ** 2 + g
            F = np.vstack([f1, f2]).T; G = bi_constraints(F, g, X)

        elif prob in ("dascmop5", "das-cmop5", "cmop5"):
            g = g_4to6(X); f1 = x1 + g; f2 = 1.0 - np.sqrt(np.maximum(x1, 0.0)) + g
            F = np.vstack([f1, f2]).T; G = bi_constraints(F, g, X)

        elif prob in ("dascmop6", "das-cmop6", "cmop6"):
            g = g_4to6(X); f1 = x1 + g
            f2 = 1.0 - np.sqrt(np.maximum(x1, 0.0)) + 0.5 * np.abs(np.sin(5 * math.pi * x1)) + g
            F = np.vstack([f1, f2]).T; G = bi_constraints(F, g, X)

        elif prob in ("dascmop7", "das-cmop7", "cmop7"):
            g = g_7to8(X)
            f1 = X[:, 0] * X[:, 1] + g
            f2 = X[:, 1] * (1.0 - X[:, 0]) + g
            f3 = 1.0 - X[:, 1] + g
            F = np.vstack([f1, f2, f3]).T; G = tri_constraints(F, g, X)

        elif prob in ("dascmop8", "das-cmop8", "cmop8"):
            g = g_7to8(X)
            f1 = np.cos(0.5 * math.pi * X[:, 0]) * np.cos(0.5 * math.pi * X[:, 1]) + g
            f2 = np.cos(0.5 * math.pi * X[:, 0]) * np.sin(0.5 * math.pi * X[:, 1]) + g
            f3 = np.sin(0.5 * math.pi * X[:, 0]) + g
            F = np.vstack([f1, f2, f3]).T; G = tri_constraints(F, g, X)

        elif prob in ("dascmop9", "das-cmop9", "cmop9"):
            g = g_9(X)
            f1 = np.cos(0.5 * math.pi * X[:, 0]) * np.cos(0.5 * math.pi * X[:, 1]) + g
            f2 = np.cos(0.5 * math.pi * X[:, 0]) * np.sin(0.5 * math.pi * X[:, 1]) + g
            f3 = np.sin(0.5 * math.pi * X[:, 0]) + g
            F = np.vstack([f1, f2, f3]).T; G = tri_constraints(F, g, X)

        else:
            raise ValueError("Unknown problem name.")

        # γ=0 -> remove obstacle-type constraints
        if gamma == 0.0:
            if F.shape[1] == 2: G[:, 2:] = 1.0
            else:               G[:, 3:] = 1.0
        return F, G

    lb = np.zeros(n_var); ub = np.ones(n_var)
    n_obj = 3 if problem in (
        "dascmop7","dascmop8","dascmop9","das-cmop7","das-cmop8","das-cmop9","cmop7","cmop8","cmop9"
    ) else 2
    return evaluate, lb, ub, n_obj

# ============================ Constraint handling & selection ============================
def total_violation(G):
    return np.sum(np.maximum(0.0, -G), axis=1)

def eps_residual(CV, eps):
    return np.maximum(0.0, CV - eps)

def eps_deb_dominates(aF, aCV, bF, bCV, eps):
    ra, rb = eps_residual(aCV, eps), eps_residual(bCV, eps)
    if ra == 0 and rb == 0:
        le = np.all(aF <= bF + 1e-12); lt = np.any(aF < bF - 1e-12)
        return le and lt
    if ra == 0 and rb > 0: return True
    if ra > 0 and rb == 0: return False
    return ra < rb

def nd_sort_eps(F: np.ndarray, CV: np.ndarray, eps: float) -> List[List[int]]:
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q: continue
            if eps_deb_dominates(F[p], CV[p], F[q], CV[q], eps):
                S[p].append(q)
            elif eps_deb_dominates(F[q], CV[q], F[p], CV[p], eps):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()
    return fronts

def crowding_distance(F: np.ndarray, idxs: List[int]) -> np.ndarray:
    if not idxs: return np.array([])
    m = F.shape[1]; l = len(idxs)
    D = np.zeros(l)
    if l <= 2:
        D[:] = np.inf; return D
    sub = F[idxs]
    for j in range(m):
        order = np.argsort(sub[:,j])
        vmin, vmax = sub[order[0],j], sub[order[-1],j]
        D[order[0]] = D[order[-1]] = np.inf
        rng = max(vmax - vmin, 1e-12)
        for k in range(1, l-1):
            D[order[k]] += (sub[order[k+1],j] - sub[order[k-1],j]) / rng
    return D

def select_env(F: np.ndarray, CV: np.ndarray, eps: float, N: int) -> List[int]:
    fronts = nd_sort_eps(F, CV, eps)
    sel = []
    for fr in fronts:
        if len(sel) + len(fr) <= N:
            sel.extend(fr)
        else:
            need = N - len(sel)
            cd = crowding_distance(F, fr)
            order = np.argsort(-cd)
            sel.extend([fr[i] for i in order[:need]])
            break
    return sel

# ============================ Variation operators ============================
def sbx_pair(p1: np.ndarray, p2: np.ndarray, lb: np.ndarray, ub: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    u = np.random.rand(*p1.shape)
    beta = np.where(u <= 0.5, (2*u)**(1/(eta+1)), (2*(1-u))**(-1/(eta+1)))
    c1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
    c2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)

def poly_mut(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, eta: float, p: float) -> np.ndarray:
    y = x.copy(); n = x.size
    for i in range(n):
        if np.random.rand() < p/n:
            u = np.random.rand()
            if u < 0.5: delta = (2*u)**(1/(eta+1)) - 1
            else:       delta = 1 - (2*(1-u))**(1/(eta+1))
            y[i] = x[i] + (ub[i]-lb[i])*delta
    return np.clip(y, lb, ub)

# ============================ E²-ε MOEA core ============================
@dataclass
class EAConfig:
    pop_size: int = 150
    n_gen: int = 200
    cross_eta: float = 15.0
    mut_eta: float = 20.0
    p_mut: float = 1.0
    # epsilon schedule
    eps_mode: str = "poly"      # 'poly' or 'exp'
    eps_p: float = 2.0          # polynomial exponent
    eps_decay: float = 0.97     # exponential ratio per gen
    eps_min_ratio: float = 1e-4 # floor = eps0 * eps_min_ratio
    # stagnation restart
    hv_tol: float = 1e-4
    stall_patience: int = 25
    restart_frac: float = 0.35
    min_feasible_ratio: float = 0.05
    # hv mc samples (for 3D)
    hv_samples: int = 20000
    # tournament
    tournament_k: int = 2

class EpsilonScheduler:
    def __init__(self, eps0:float, n_gen:int, mode:str="poly", p:float=2.0, decay:float=0.97, min_ratio:float=1e-4):
        self.eps0 = float(max(eps0, 1e-16))
        self.T = int(n_gen)
        self.mode = mode
        self.p = float(p)
        self.decay = float(decay)
        self.floor = float(min_ratio) * self.eps0
    def at(self, t:int) -> float:
        t = min(max(t,0), self.T)
        if self.mode == "poly": val = self.eps0 * ((1 - t/self.T)**self.p)
        else:                   val = self.eps0 * (self.decay**t)
        return float(max(val, self.floor))

# ---------- HV & IGD ----------
def hv_2d(front: np.ndarray, ref: np.ndarray) -> float:
    """Exact HV in 2D (minimization). Assumes ref >= front component-wise."""
    if front.size == 0: return 0.0
    # keep ND
    F = []
    for i in range(front.shape[0]):
        fi = front[i]
        dominated = np.any(np.all(front <= fi, axis=1) & np.any(front < fi, axis=1))
        if not dominated: F.append(fi)
    F = np.array(F)
    F = F[np.argsort(F[:,0])]
    hv = 0.0
    prev_f1 = ref[0]
    for i in range(F.shape[0]-1, -1, -1):
        f1, f2 = F[i,0], F[i,1]
        width  = prev_f1 - f1
        height = max(0.0, ref[1] - f2)
        hv += max(0.0, width) * height
        prev_f1 = f1
    return float(max(hv, 0.0))

def hv_3d_mc(front: np.ndarray, ref: np.ndarray, samples:int=20000, rng=None) -> float:
    """Monte Carlo HV estimate in 3D (minimization)."""
    if front.size == 0: return 0.0
    rng = np.random.default_rng() if rng is None else rng
    mins = np.min(front, axis=0)
    box_min = mins
    box_max = ref
    vol_box = np.prod(box_max - box_min)
    if vol_box <= 0: return 0.0
    U = rng.random((samples, 3)) * (box_max - box_min) + box_min
    covered = np.any(np.all(front[None,:,:] <= U[:,None,:], axis=2), axis=1)
    frac = np.mean(covered)
    return float(frac * vol_box)

def hv_generic(F: np.ndarray, ref: np.ndarray, hv_samples:int=20000) -> float:
    m = F.shape[1]
    if m == 2: return hv_2d(F, ref)
    if m == 3: return hv_3d_mc(F, ref, samples=hv_samples)
    return float("nan")

def load_ref_pf(problem_id:int) -> Optional[np.ndarray]:
    folder = "ref"
    cand = [
        os.path.join(folder, f"DASCMOP{problem_id}.pf"),
        os.path.join(folder, f"DASCMOP{problem_id}.csv"),
        os.path.join(folder, f"DAS-CMOP{problem_id}.pf"),
        os.path.join(folder, f"DAS-CMOP{problem_id}.csv"),
    ]
    for p in cand:
        if os.path.exists(p):
            try:
                arr = np.loadtxt(p, delimiter=',' if p.endswith(".csv") else None)
                arr = np.atleast_2d(arr).astype(float)
                if arr.shape[1] in (2,3): return arr
            except Exception:
                pass
    return None

def igd(P: np.ndarray, R: np.ndarray) -> float:
    """IGD(P,R): average distance from each ref r∈R to nearest p∈P."""
    if P.size == 0 or R is None or R.size == 0: return np.inf
    d = np.sqrt(((R[:,None,:] - P[None,:,:])**2).sum(axis=2)).min(axis=1)
    return float(np.mean(d))

# ---------- Core algorithm ----------
def e2_eps_moea(
    evaluate: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    lb: np.ndarray, ub: np.ndarray, n_obj: int, n_var: int,
    cfg: EAConfig, seed:int, log_dir:str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    rng = np.random.default_rng(seed)
    xl, xu = np.array(lb), np.array(ub)

    # LHS init
    X01 = lhs(cfg.pop_size, n_var, rng)
    X = X01 * (xu - xl) + xl

    def eval_pop(P):
        F, G = evaluate(P)
        return F, total_violation(G)

    F, CV = eval_pop(X)
    eps0 = float(np.percentile(CV, 75)) if np.any(CV > 0) else 1e-6
    scheduler = EpsilonScheduler(eps0, cfg.n_gen, cfg.eps_mode, cfg.eps_p, cfg.eps_decay, cfg.eps_min_ratio)

    ref_pf = None  # filled in runner if available
    global_ref = None
    logs = []
    restart_marks = []

    best_hv = -1.0
    last_impr = 0

    def dyn_ref(Fall: np.ndarray):
        mx = np.max(Fall, axis=0)
        return mx + 0.1*np.abs(mx) + 1e-9

    for gen in range(1, cfg.n_gen+1):
        eps = scheduler.at(gen)

        # parents: eps-tournament
        parents_idx = []
        for _ in range(cfg.pop_size):
            cand = rng.integers(0, cfg.pop_size, size=cfg.tournament_k)
            a, b = cand[0], cand[1]
            better = eps_deb_dominates(F[a], CV[a], F[b], CV[b], eps)
            parents_idx.append(a if better else b)
        P = X[parents_idx]

        # variation
        C = []
        phase = gen / cfg.n_gen
        mut_eta = cfg.mut_eta * (1.0 + 2.0*(1.0 - phase))  # early: larger, late: smaller
        for i in range(0, cfg.pop_size, 2):
            c1, c2 = sbx_pair(P[i], P[(i+1) % cfg.pop_size], xl, xu, cfg.cross_eta)
            c1 = poly_mut(c1, xl, xu, mut_eta, cfg.p_mut)
            c2 = poly_mut(c2, xl, xu, mut_eta, cfg.p_mut)
            C.append(c1); C.append(c2)
        C = np.array(C[:cfg.pop_size])

        F_C, CV_C = eval_pop(C)
        X_all = np.vstack([X, C])
        F_all = np.vstack([F, F_C])
        CV_all = np.hstack([CV, CV_C])

        sel = select_env(F_all, CV_all, eps, cfg.pop_size)
        X, F, CV = X_all[sel], F_all[sel], CV_all[sel]

        # update global_ref if no true reference
        if ref_pf is None:
            if global_ref is None: global_ref = F.copy()
            else:
                R = np.vstack([global_ref, F])
                keep = []
                for i in range(R.shape[0]):
                    dominated = np.any(np.all(R <= R[i], axis=1) & np.any(R < R[i], axis=1))
                    keep.append(not dominated)
                global_ref = R[np.array(keep)]

        # metrics
        ref_box = dyn_ref(F)
        hv = hv_generic(F, ref_box, hv_samples=cfg.hv_samples)
        igd_val = igd(F, ref_pf if ref_pf is not None else global_ref)
        feas_ratio = float(np.mean(CV <= eps))
        logs.append(dict(gen=gen, eps=eps, hv=hv, igd=igd_val,
                         feasible_ratio=feas_ratio, pop_feasible=int(np.sum(CV <= eps))))

        # stagnation / restart
        improved = hv > best_hv + cfg.hv_tol
        if improved:
            best_hv = hv; last_impr = gen
        need_restart = (gen - last_impr) >= cfg.stall_patience or (feas_ratio < cfg.min_feasible_ratio and gen > 20)
        if need_restart:
            restart_marks.append((gen, hv))
            k = int(cfg.restart_frac * cfg.pop_size)
            elite_idx = select_env(F, CV, eps, cfg.pop_size - k)
            elites = X[elite_idx]
            rand = xl + (xu - xl) * rng.random((k, n_var))
            X = np.vstack([elites, rand])
            F, CV = eval_pop(X)
            last_impr = gen  # reset

    with open(os.path.join(log_dir, "restart_marks.json"), "w", encoding="utf-8") as f:
        json.dump([{"gen":g, "hv":float(h)} for g, h in restart_marks], f, indent=2)

    return X, F, CV, logs

# ============================ Runner (run mode) ============================
def run_one_problem(
    pid:int, n_var:int, triplet:Tuple[float,float,float],
    cfg:EAConfig, seed:int, out_root:str
):
    set_seed(seed)
    name = f"DASCMOP{pid}"
    run_dir = os.path.join(out_root, name, f"seed_{seed}")
    ensure_dir(run_dir)
    print(f"[{now()}] >>> {name} | seed={seed} | out={run_dir}")

    evaluate, lb, ub, n_obj = build_dascmop(name, n_var, triplet)
    X, F, CV, logs = e2_eps_moea(evaluate, lb, ub, n_obj, n_var, cfg, seed, run_dir)

    save_csv(os.path.join(run_dir, "final_X.csv"), X)
    save_csv(os.path.join(run_dir, "final_F.csv"), F)
    save_csv(os.path.join(run_dir, "final_CV.csv"), CV.reshape(-1,1))

    with open(os.path.join(run_dir, "progress.csv"), "w", encoding="utf-8") as f:
        f.write("gen,eps,hv,igd,feasible_ratio,pop_feasible\n")
        for r in logs:
            f.write("{},{:.12g},{:.12g},{:.12g},{:.6f},{}\n".format(
                r["gen"], r["eps"], r["hv"], r["igd"], r["feasible_ratio"], r["pop_feasible"]
            ))

    meta = dict(problem=name, n_obj=int(n_obj), n_var=int(n_var),
                triplet=dict(eta=triplet[0], zeta=triplet[1], gamma=triplet[2]),
                seed=int(seed), cfg=cfg.__dict__,
                note="IGD uses provided ref PF if ./ref/ exists; otherwise global ND archive approximation.")
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    gens = [r["gen"] for r in logs]
    epss = [r["eps"] for r in logs]
    hv_list = [r["hv"] for r in logs]
    igd_list = [r["igd"] for r in logs]

    try:
        with open(os.path.join(run_dir, "restart_marks.json"), "r", encoding="utf-8") as f:
            rmarks = json.load(f)
    except Exception:
        rmarks = []

    plt.figure(figsize=(8,4.5))
    plt.plot(gens, epss, lw=2)
    for m in rmarks:
        plt.axvline(m["gen"], color="crimson", ls="--", lw=1)
    plt.xlabel("Generation"); plt.ylabel("Epsilon")
    plt.title(f"{name} ε-schedule (seed={seed})  —  restarts marked")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "epsilon_schedule.png"), dpi=150)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8,4.5))
    ax1.plot(gens, hv_list, lw=2, label="HV")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("HV")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(gens, igd_list, lw=2, color="tab:orange", label="IGD")
    ax2.set_ylabel("IGD")
    lines, labels = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lb2, loc="best")
    plt.title(f"{name} HV & IGD (seed={seed})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "hv_igd.png"), dpi=150)
    plt.close()

    if F.shape[1] == 2:
        plt.figure(figsize=(5.5,5))
        plt.scatter(F[:,0], F[:,1], s=12)
        plt.xlabel("f1"); plt.ylabel("f2")
        plt.title(f"{name} final nondominated pop (seed={seed})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "final_front.png"), dpi=150)
        plt.close()
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(F[:,0], F[:,1], F[:,2], s=10)
        ax.set_xlabel("f1"); ax.set_ylabel("f2"); ax.set_zlabel("f3")
        ax.set_title(f"{name} final nondominated pop (seed={seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "final_front.png"), dpi=150)
        plt.close()

    print(f"[{now()}] <<< Done {name} | seed={seed} | log saved to progress.csv")

def parse_problem_list(spec: str) -> List[int]:
    spec = spec.strip()
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(list(range(int(a), int(b)+1)))
        else:
            ids.append(int(part))
    ids = sorted(list(set([i for i in ids if 1 <= i <= 9])))
    if not ids: raise ValueError("Empty problem set.")
    return ids

# ============================ Analyzer (analyze mode) ============================
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(w) / w
    return np.convolve(xp, ker, mode="valid")

def read_progress_csv(path: str) -> Dict[str, np.ndarray]:
    gens, eps, hv, igd, feas, popf = [], [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                gens.append(int(row["gen"]))
                eps.append(float(row["eps"]))
                hv.append(float(row["hv"]))
                igd.append(float(row["igd"]))
                feas.append(float(row["feasible_ratio"]))
                popf.append(float(row["pop_feasible"]))
            except Exception:
                continue
    if not gens:
        raise ValueError(f"No data in {path}")
    order = np.argsort(gens)
    return dict(
        gen=np.array(gens, dtype=int)[order],
        eps=np.array(eps, dtype=float)[order],
        hv=np.array(hv, dtype=float)[order],
        igd=np.array(igd, dtype=float)[order],
        feasible_ratio=np.array(feas, dtype=float)[order],
        pop_feasible=np.array(popf, dtype=float)[order],
    )

def align_by_gen(series_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    all_gens = sorted(set(int(g) for s in series_list for g in s["gen"]))
    G = np.array(all_gens, dtype=int)
    def align_one(key: str) -> np.ndarray:
        M = np.full((len(series_list), len(G)), np.nan, dtype=float)
        for i, s in enumerate(series_list):
            g = s["gen"]; idx = {int(v): j for j, v in enumerate(g)}
            for col, gl in enumerate(G):
                j = idx.get(int(gl), None)
                if j is not None:
                    M[i, col] = float(s[key][j])
        return M
    out = {"gen": G}
    for key in ["eps", "hv", "igd", "feasible_ratio", "pop_feasible"]:
        out[key] = align_one(key)
    return out

def mean_std_ignore_nan(M: np.ndarray):
    count = np.sum(~np.isnan(M), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(M, axis=0)
        std  = np.nanstd(M, axis=0, ddof=0)
    return mean, std, count

def plot_mean_std(x, mean, std, title, xlabel, ylabel, out_path, smooth=0):
    if smooth and smooth > 1:
        mean_plot = moving_average(mean, smooth)
        std_plot  = moving_average(std, smooth)
        x_plot    = moving_average(np.asarray(x, dtype=float), smooth)
    else:
        mean_plot, std_plot, x_plot = mean, std, x
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(x_plot, mean_plot, lw=2)
    y1 = mean_plot - std_plot
    y2 = mean_plot + std_plot
    plt.fill_between(x_plot, y1, y2, alpha=0.25)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def process_problem_analysis(root: str, pid: int, out_root: str, smooth: int, min_seeds: int) -> bool:
    prob_dir = os.path.join(root, f"DASCMOP{pid}")
    seed_dirs = sorted(glob.glob(os.path.join(prob_dir, "seed_*")))
    if not seed_dirs:
        print(f"[WARN] No seeds found for DASCMOP{pid} under {prob_dir}")
        return False

    series = []
    for sd in seed_dirs:
        fcsv = os.path.join(sd, "progress.csv")
        if not os.path.exists(fcsv): continue
        try:
            s = read_progress_csv(fcsv)
            series.append(s)
        except Exception as e:
            print(f"[WARN] Skip {fcsv}: {e}")

    if len(series) < max(1, min_seeds):
        print(f"[WARN] Not enough valid seeds for DASCMOP{pid} (got {len(series)})")
        return False

    aligned = align_by_gen(series)
    G = aligned["gen"]
    hv_mean, hv_std, hv_cnt = mean_std_ignore_nan(aligned["hv"])
    igd_mean, igd_std, igd_cnt = mean_std_ignore_nan(aligned["igd"])
    feas_mean, feas_std, feas_cnt = mean_std_ignore_nan(aligned["feasible_ratio"])
    eps_mean, eps_std, eps_cnt = mean_std_ignore_nan(aligned["eps"])

    out_dir = os.path.join(out_root, f"DASCMOP{pid}")
    ensure_dir(out_dir)
    agg_csv = os.path.join(out_dir, "agg_progress.csv")
    with open(agg_csv, "w", encoding="utf-8") as f:
        f.write("gen,hv_mean,hv_std,igd_mean,igd_std,feasible_mean,feasible_std,eps_mean,eps_std,count\n")
        for i in range(len(G)):
            cnt = int(max(hv_cnt[i], igd_cnt[i], feas_cnt[i], eps_cnt[i]))
            f.write("{},{:.12g},{:.12g},{:.12g},{:.12g},{:.6f},{:.6f},{:.12g},{:.12g},{}\n".format(
                int(G[i]),
                hv_mean[i], hv_std[i],
                igd_mean[i], igd_std[i],
                feas_mean[i], feas_std[i],
                eps_mean[i], eps_std[i],
                cnt
            ))

    plot_mean_std(G, hv_mean, hv_std,
                  title=f"DASCMOP{pid}  HV  (mean ± std, n={len(series)})",
                  xlabel="Generation", ylabel="HV",
                  out_path=os.path.join(out_dir, "hv_mean_std.png"),
                  smooth=smooth)
    plot_mean_std(G, igd_mean, igd_std,
                  title=f"DASCMOP{pid}  IGD  (mean ± std, n={len(series)})",
                  xlabel="Generation", ylabel="IGD",
                  out_path=os.path.join(out_dir, "igd_mean_std.png"),
                  smooth=smooth)
    plot_mean_std(G, feas_mean, feas_std,
                  title=f"DASCMOP{pid}  Feasible Ratio  (mean ± std, n={len(series)})",
                  xlabel="Generation", ylabel="Feasible Ratio",
                  out_path=os.path.join(out_dir, "feasible_ratio_mean_std.png"),
                  smooth=smooth)

    print(f"[OK] DASCMOP{pid}: seeds={len(series)} | saved -> {out_dir}")
    return True

def overlay_summary(out_root: str, pids: List[int], which: str, smooth: int):
    """Overlay mean curves across problems (no std bands) for quick comparison."""
    plt.figure(figsize=(8.5, 5.2))
    legend = []
    for pid in pids:
        agg_csv = os.path.join(out_root, f"DASCMOP{pid}", "agg_progress.csv")
        if not os.path.exists(agg_csv):
            continue
        data = np.genfromtxt(agg_csv, delimiter=",", names=True, dtype=None, encoding=None)
        x = data["gen"]
        y = data["hv_mean"] if which == "hv" else data["igd_mean"]
        if smooth and smooth > 1:
            x = moving_average(x.astype(float), smooth)
            y = moving_average(y.astype(float), smooth)
        plt.plot(x, y, lw=2)
        legend.append(f"P{pid}")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Generation"); plt.ylabel(which.upper())
    plt.title(f"Overlay of {which.upper()} means across problems")
    if legend:
        plt.legend(legend, ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"summary_{which}.png"), dpi=300)
    plt.close()
    if legend:
        print(f"[OK] Summary overlay saved: {os.path.join(out_root, f'summary_{which}.png')}")
    else:
        print(f"[WARN] Summary overlay skipped (no aggregated files).")

# ============================ CLI (run / analyze / both) ============================
def main():
    ap = argparse.ArgumentParser(description="All-in-one E²-ε MOEA (DAS-CMOP1..9) + Analyzer")
    ap.add_argument("--mode", type=str, default="run", choices=["run","analyze","both"],
                    help="run: execute experiments; analyze: aggregate; both: run then analyze")
    # run args
    ap.add_argument("--problems", type=str, default="1-9", help="e.g. '1-3,5,9' or '1-9'")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--gen", type=int, default=300)
    ap.add_argument("--nvar", type=int, default=30)
    ap.add_argument("--triplet", type=str, default="0.5,0.5,0.5", help="eta,zeta,gamma in [0,1]")
    # epsilon schedule
    ap.add_argument("--eps_mode", type=str, default="poly", choices=["poly","exp"])
    ap.add_argument("--eps_p", type=float, default=2.0)
    ap.add_argument("--eps_decay", type=float, default=0.97)
    ap.add_argument("--eps_min_ratio", type=float, default=1e-4)
    # restarts & hv
    ap.add_argument("--stall_patience", type=int, default=25)
    ap.add_argument("--hv_tol", type=float, default=1e-4)
    ap.add_argument("--restart_frac", type=float, default=0.35)
    ap.add_argument("--min_feasible_ratio", type=float, default=0.05)
    ap.add_argument("--hv_samples", type=int, default=20000, help="for 3D HV Monte Carlo")
    # opers
    ap.add_argument("--sbx_eta", type=float, default=15.0)
    ap.add_argument("--mut_eta", type=float, default=20.0)
    ap.add_argument("--p_mut", type=float, default=1.0, help="poly mutation prob per-solution (actual per-dim = p_mut/nvar)")
    ap.add_argument("--out", type=str, default="./runs_e2_eps_dascmop_single")
    # analyze args
    ap.add_argument("--analysis_root", type=str, default=None, help="If empty, use --out")
    ap.add_argument("--analysis_problems", type=str, default=None, help="If empty, use --problems")
    ap.add_argument("--analysis_out", type=str, default="analysis", help="Output folder for aggregation")
    ap.add_argument("--smooth", type=int, default=0, help="Moving average window size (0/1 = no smoothing)")
    ap.add_argument("--min_seeds", type=int, default=1, help="Require at least this many valid seeds per problem")
    ap.add_argument("--summary", action="store_true", help="Also draw overlays of HV and IGD means across problems")
    args = ap.parse_args()

    eta, zeta, gamma = [float(x) for x in args.triplet.split(",")]
    cfg = EAConfig(
        pop_size=args.pop, n_gen=args.gen,
        cross_eta=args.sbx_eta, mut_eta=args.mut_eta, p_mut=args.p_mut,
        eps_mode=args.eps_mode, eps_p=args.eps_p, eps_decay=args.eps_decay, eps_min_ratio=args.eps_min_ratio,
        hv_tol=args.hv_tol, stall_patience=args.stall_patience,
        restart_frac=args.restart_frac, min_feasible_ratio=args.min_feasible_ratio,
        hv_samples=args.hv_samples
    )

    if args.mode in ("run", "both"):
        ensure_dir(args.out)
        with open(os.path.join(args.out, "cmdline.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

        ids = parse_problem_list(args.problems)
        for pid in ids:
            for s in range(args.seeds):
                seed = 1000 + 13*pid + 7*s
                run_one_problem(pid, args.nvar, (eta, zeta, gamma), cfg, seed, args.out)
        print(f"[{now()}] Run mode done. Outputs in: {os.path.abspath(args.out)}")

    if args.mode in ("analyze", "both"):
        root = args.analysis_root or args.out
        p_spec = args.analysis_problems or args.problems
        pids = parse_problem_list(p_spec)
        os.makedirs(args.analysis_out, exist_ok=True)

        any_ok = False
        for pid in pids:
            ok = process_problem_analysis(root, pid, args.analysis_out, args.smooth, args.min_seeds)
            any_ok = any_ok or ok
        if args.summary and any_ok:
            overlay_summary(args.analysis_out, pids, which="hv",  smooth=args.smooth)
            overlay_summary(args.analysis_out, pids, which="igd", smooth=args.smooth)
        print(f"[{now()}] Analyze mode done. Outputs in: {os.path.abspath(args.analysis_out)}")

if __name__ == "__main__":
    main()
