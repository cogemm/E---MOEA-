import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. Base directory = folder where this script lives
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Root folders for each method (relative to BASE_DIR)
METHOD_ROOTS = {
    "NSGA2_CDP":         "runs_nsga2_cdp",
    "E2_eps_no_restart": "runs_e2_eps_no_restart",
    "E2_eps_static_eps": "runs_e2_eps_static_eps",
    "E2_eps_full":       "runs_e2_eps_dascmop_single",
}

# 2. Labels and styles for plotting
METHOD_ORDER = ["NSGA2_CDP", "E2_eps_no_restart", "E2_eps_static_eps", "E2_eps_full"]

METHOD_LABELS = {
    "NSGA2_CDP":         "NSGA-II (CDP)",
    "E2_eps_no_restart": r"E$^2$-$\epsilon$ (no restart)",
    "E2_eps_static_eps": r"E$^2$-$\epsilon$ (static $\epsilon$)",
    "E2_eps_full":       r"E$^2$-$\epsilon$ (full)",
}

METHOD_STYLES = {
    "NSGA2_CDP":         dict(marker="o", linestyle="-"),
    "E2_eps_no_restart": dict(marker="s", linestyle="--"),
    "E2_eps_static_eps": dict(marker="^", linestyle="-."),
    "E2_eps_full":       dict(marker="D", linestyle="-"),
}

# 3. Problems DAS-CMOP1 ~ DAS-CMOP9
PROBLEMS = list(range(1, 10))


def summarize_one_method(problem_id, method_key):
    """
    For a given problem and method, read all seed_*/progress.csv
    and return mean/std of FINAL HV and IGD.

    Returns:
        (hv_mean, hv_std, igd_mean, igd_std, n_runs)
        or (None, None, None, None, 0) if nothing found.
    """
    root_dir = os.path.join(BASE_DIR, METHOD_ROOTS[method_key])
    problem_dir = os.path.join(root_dir, f"DASCMOP{problem_id}")

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

        # data may be 0-D or 1-D structured array; np.atleast_1d makes it indexable
        data = np.atleast_1d(data)

        hv = data["hv"].astype(float)
        igd = data["igd"].astype(float)

        # use the LAST generation as the final value
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


def plot_figure3(out_path="figure3_ablation_dascmop1_9.png"):
    """
    Figure 3: Ablation study on DAS-CMOP1–9.

    Left panel: HV vs. problem index (1–9) with mean ± std errorbars for each method.
    Right panel: IGD vs. problem index (1–9) with mean ± std errorbars for each method.
    """
    # containers: method -> arrays over problems
    hv_mean = {m: [] for m in METHOD_ORDER}
    hv_std  = {m: [] for m in METHOD_ORDER}
    igd_mean = {m: [] for m in METHOD_ORDER}
    igd_std  = {m: [] for m in METHOD_ORDER}

    for pid in PROBLEMS:
        print(f"[INFO] summarizing DAS-CMOP{pid}...")
        for method_key in METHOD_ORDER:
            h_mean, h_std, i_mean, i_std, n_runs = summarize_one_method(pid, method_key)
            if n_runs == 0:
                # fill with NaN to keep indices aligned
                hv_mean[method_key].append(np.nan)
                hv_std[method_key].append(0.0)
                igd_mean[method_key].append(np.nan)
                igd_std[method_key].append(0.0)
            else:
                hv_mean[method_key].append(h_mean)
                hv_std[method_key].append(h_std)
                igd_mean[method_key].append(i_mean)
                igd_std[method_key].append(i_std)

    # convert to numpy arrays
    for method_key in METHOD_ORDER:
        hv_mean[method_key] = np.array(hv_mean[method_key], dtype=float)
        hv_std[method_key]  = np.array(hv_std[method_key], dtype=float)
        igd_mean[method_key] = np.array(igd_mean[method_key], dtype=float)
        igd_std[method_key]  = np.array(igd_std[method_key], dtype=float)

    # ---------------- plotting ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # left: HV vs problem
    ax_hv = axes[0]
    for method_key in METHOD_ORDER:
        style = METHOD_STYLES[method_key]
        ax_hv.errorbar(
            PROBLEMS,
            hv_mean[method_key],
            yerr=hv_std[method_key],
            label=METHOD_LABELS[method_key],
            capsize=3,
            **style
        )
    ax_hv.set_xlabel("Problem (DAS-CMOP index)")
    ax_hv.set_ylabel("Final HV")
    ax_hv.set_xticks(PROBLEMS)
    ax_hv.grid(True, alpha=0.3)
    ax_hv.set_title("Ablation: final HV on DAS-CMOP1–9")

    # right: IGD vs problem
    ax_igd = axes[1]
    for method_key in METHOD_ORDER:
        style = METHOD_STYLES[method_key]
        ax_igd.errorbar(
            PROBLEMS,
            igd_mean[method_key],
            yerr=igd_std[method_key],
            label=METHOD_LABELS[method_key],
            capsize=3,
            **style
        )
    ax_igd.set_xlabel("Problem (DAS-CMOP index)")
    ax_igd.set_ylabel("Final IGD")
    ax_igd.set_xticks(PROBLEMS)
    ax_igd.grid(True, alpha=0.3)
    ax_igd.set_title("Ablation: final IGD on DAS-CMOP1–9")

    # shared legend
    handles, labels = ax_hv.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Figure 3 saved to: {out_path}")


if __name__ == "__main__":
    plot_figure3()
