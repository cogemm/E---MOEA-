import os
import glob
import numpy as np

# ---------------------------------------------------------
# 0. Base directory = folder where this script lives
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Where each method's runs are stored
METHOD_ROOTS = {
    "E2_eps_full":       "runs_e2_eps_dascmop_single",
    "E2_eps_no_restart": "runs_e2_eps_no_restart",
    "E2_eps_static_eps": "runs_e2_eps_static_eps",
    "NSGA2_CDP":         "runs_nsga2_cdp",
}

# 2. Problems DAS-CMOP1 ~ DAS-CMOP9
PROBLEMS = list(range(1, 10))


def summarize_one_method(problem_id, method_key):
    """
    For a given problem and method, read all seed_*/progress.csv
    and return mean/std of FINAL HV and IGD.

    Returns:
        (hv_mean, hv_std, igd_mean, igd_std, n_runs)
        or (None, None, None, None, 0) if nothing found.
    """
    root = os.path.join(BASE_DIR, METHOD_ROOTS[method_key])
    problem_dir = os.path.join(root, f"DASCMOP{problem_id}")

    seed_dirs = sorted(glob.glob(os.path.join(problem_dir, "seed_*")))
    if not seed_dirs:
        print(f"[WARN] {method_key} – DAS-CMOP{problem_id}: no seed_* folders.")
        return None, None, None, None, 0

    hv_final = []
    igd_final = []

    for sd in seed_dirs:
        prog_path = os.path.join(sd, "progress.csv")
        if not os.path.isfile(prog_path):
            print(f"[WARN] {method_key} – {sd}: progress.csv not found, skip.")
            continue

        try:
            data = np.genfromtxt(prog_path, delimiter=",", names=True)
        except Exception as e:
            print(f"[WARN] {method_key} – failed to read {prog_path}: {e}")
            continue

        # data may be a 0-D or 1-D structured array; use np.atleast_1d
        data = np.atleast_1d(data)

        hv = data["hv"].astype(float)
        igd = data["igd"].astype(float)

        hv_final.append(hv[-1])
        igd_final.append(igd[-1])

    hv_final = np.array(hv_final, dtype=float)
    igd_final = np.array(igd_final, dtype=float)

    n_runs = hv_final.size
    if n_runs == 0:
        print(f"[WARN] {method_key} – DAS-CMOP{problem_id}: no valid runs.")
        return None, None, None, None, 0

    hv_mean = hv_final.mean()
    hv_std = hv_final.std()
    igd_mean = igd_final.mean()
    igd_std = igd_final.std()

    return hv_mean, hv_std, igd_mean, igd_std, n_runs


def main():
    print("===== Table 2 summary (final HV and IGD for all DAS-CMOP1–9) =====")
    print("Each cell: mean_HV ± std_HV  (mean_IGD ± std_IGD)")

    # Option 1: print nice LaTeX-style table rows
    header_methods = [
        "NSGA-II (CDP)",
        r"E$^2$-$\epsilon$ (no restart)",
        r"E$^2$-$\epsilon$ (static $\epsilon$)",
        r"E$^2$-$\epsilon$ (full)",
    ]
    print("\nLaTeX rows (one per problem):\n")

    for pid in PROBLEMS:
        row_cells = []
        for method_key in ["NSGA2_CDP", "E2_eps_no_restart", "E2_eps_static_eps", "E2_eps_full"]:
            hv_mean, hv_std, igd_mean, igd_std, n_runs = summarize_one_method(pid, method_key)
            if n_runs == 0:
                cell = "–"
            else:
                cell = f"{hv_mean:.3g}±{hv_std:.2g} ({igd_mean:.3g}±{igd_std:.2g})"
            row_cells.append(cell)

        # Example LaTeX row:
        # DAS-CMOP1 & 0.12±0.03 (0.8±0.1) & ... \\
        latex_row = "DAS-CMOP%d & %s \\\\" % (pid, " & ".join(row_cells))
        print(latex_row)

    # Option 2: also print a verbose version for checking
    print("\nVerbose summary:\n")
    for pid in PROBLEMS:
        print(f"--- DAS-CMOP{pid} ---")
        for method_key, root in METHOD_ROOTS.items():
            hv_mean, hv_std, igd_mean, igd_std, n_runs = summarize_one_method(pid, method_key)
            if n_runs == 0:
                print(f"{method_key:16s}: no runs")
            else:
                print(
                    f"{method_key:16s}: "
                    f"HV = {hv_mean:.4g} ± {hv_std:.4g}, "
                    f"IGD = {igd_mean:.4g} ± {igd_std:.4g} "
                    f"(n={n_runs})"
                )
        print()

if __name__ == "__main__":
    main()
