import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1) 基本路径设置：用脚本所在目录作为 BASE_DIR
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# full E²-ε 结果所在的根目录
RUN_ROOT = os.path.join(BASE_DIR, "runs_e2_eps_dascmop_single")

# 想画哪些 DAS-CMOP（与你论文 4.2 中的描述对应即可）
# 你可以改成 [1, 4, 6, 8] 或 [2, 3, 7, 9] 等
PROBLEMS = [1,2, 3,4,5,6, 7, 8,9]


def load_progress_for_problem(problem_id):
    """
    读取某个 DAS-CMOP 下所有 seed 的 progress.csv，
    返回：
      gens: 代数向量 (长度 T)
      hv_arr: 形状 (num_seeds, T) 的 HV 数组
      igd_arr: 形状 (num_seeds, T) 的 IGD 数组
    """
    problem_dir = os.path.join(RUN_ROOT, f"DASCMOP{problem_id}")
    seed_dirs = sorted(glob.glob(os.path.join(problem_dir, "seed_*")))

    hv_list = []
    igd_list = []
    gens_ref = None

    if not seed_dirs:
        raise RuntimeError(f"No seed_* folders found in {problem_dir}")

    for sd in seed_dirs:
        prog_path = os.path.join(sd, "progress.csv")
        if not os.path.isfile(prog_path):
            print(f"[WARN] {prog_path} not found, skip.")
            continue

        # 读取 CSV（第一行是表头）
        data = np.genfromtxt(prog_path, delimiter=",", names=True)

        gens = data["gen"].astype(int)
        hv = data["hv"].astype(float)
        igd = data["igd"].astype(float)

        # 确保每个 seed 的代数长度一致
        if gens_ref is None:
            gens_ref = gens
        else:
            if len(gens) != len(gens_ref):
                print(f"[WARN] {sd} has different length, skip.")
                continue

        hv_list.append(hv)
        igd_list.append(igd)

    hv_arr = np.vstack(hv_list)   # (num_seeds, T)
    igd_arr = np.vstack(igd_list) # (num_seeds, T)

    print(f"[INFO] DAS-CMOP{problem_id}: {hv_arr.shape[0]} seeds, {hv_arr.shape[1]} generations.")

    return gens_ref, hv_arr, igd_arr


def plot_convergence_figure(out_path="figure2_convergence_e2eps.png"):
    """
    绘制 Figure 2：
      - 每个问题一个子图（这里是 2×2）
      - 左轴 HV（平均±标准差）
      - 右轴 IGD（平均±标准差）
    """
    n_prob = len(PROBLEMS)
    n_rows, n_cols = 3, 3  # 这里根据 PROBLEMS 个数自己调整

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=True)
    axes = axes.ravel()

    for ax, pid in zip(axes, PROBLEMS):
        gens, hv_arr, igd_arr = load_progress_for_problem(pid)

        # 按行（seed）求平均和标准差
        hv_mean = hv_arr.mean(axis=0)
        hv_std = hv_arr.std(axis=0)
        igd_mean = igd_arr.mean(axis=0)
        igd_std = igd_arr.std(axis=0)

        # ---------------- 左轴：HV ----------------
        line_hv = ax.plot(
            gens, hv_mean,
            label="HV (mean)",
            linestyle="-"
        )[0]
        ax.fill_between(
            gens,
            hv_mean - hv_std,
            hv_mean + hv_std,
            alpha=0.2
        )
        ax.set_ylabel("HV")

        # ---------------- 右轴：IGD ----------------
        ax2 = ax.twinx()
        line_igd = ax2.plot(
            gens, igd_mean,
            label="IGD (mean)",
            linestyle="--"
        )[0]
        ax2.fill_between(
            gens,
            igd_mean - igd_std,
            igd_mean + igd_std,
            alpha=0.2
        )
        ax2.set_ylabel("IGD")

        # 标题 & 坐标轴
        ax.set_title(f"DAS-CMOP{pid}")
        ax.set_xlabel("Generation")

        # 合并左右轴的图例
        lines = [line_hv, line_igd]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8)

        ax.grid(True, alpha=0.3)

    # 如果 PROBLEMS 少于 4，可以把多余子图关掉
    for j in range(len(PROBLEMS), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Convergence of E$^2$-$\\epsilon$ MOEA (HV & IGD vs. Generations)", y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Figure 2 saved to: {out_path}")


if __name__ == "__main__":
    plot_convergence_figure()
