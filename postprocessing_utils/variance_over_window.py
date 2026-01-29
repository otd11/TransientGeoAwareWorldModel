import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm


# ============================================================
# Statistics
# ============================================================
def compute_temporal_mean_and_ci(data, alpha=0.05):
    """
    Mean and CI across samples.
    data: (N_samples, T)
    """
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    n = np.sum(~np.isnan(data), axis=0)

    z = norm.ppf(1 - alpha / 2)
    half_width = z * std / np.sqrt(n)

    lower = mean - half_width
    upper = mean + half_width
    return mean, lower, upper


# ============================================================
# Core computation
# ============================================================
def compute_spatially_averaged_temporal_variance(window, max_lag):
    """
    window: (T, H, W)
    Returns: (max_lag,)
    """
    T, H, W = window.shape
    vars_lag = []

    for lag in range(max_lag):
        segment = window[:T - lag]
        if segment.shape[0] < 2:
            vars_lag.append(np.nan)
            continue

        var_hw = np.nanvar(segment, axis=0)
        vars_lag.append(np.nanmean(var_hw))

    return np.array(vars_lag)


# ============================================================
# Main evaluation (SIM + GAN)
# ============================================================
def evaluate_temporal_variance(
    sim_tensors,
    gan_tensors,
    N_window,
    save_dir=None,
    added_info="",
    max_lag=None,
    n_samples=100,
    seed=42,
    all_windows=True,
    combine_channels=False,
    video_name=None,
    gan_ci=True
):
    """
    Temporal variance vs relative lag.

    sim_tensors: (C, F_sim, H, W)
    gan_tensors: (C, S, F_gan, H, W)
    """

    if not isinstance(sim_tensors, np.ndarray):
        sim_tensors = sim_tensors.cpu().numpy()
    if not isinstance(gan_tensors, np.ndarray):
        gan_tensors = gan_tensors.cpu().numpy()

    _, F_sim, H, W = sim_tensors.shape
    C, S, F_gan, _, _ = gan_tensors.shape

    if max_lag is None:
        max_lag = N_window

    max_start_sim = F_sim - N_window + 1
    max_start_gan = F_gan - N_window + 1

    if max_start_sim < 1 or max_start_gan < 1:
        raise ValueError("N_window larger than available frames.")

    if all_windows:
        n_samples_sim = max_start_sim
        n_samples_gan = max_start_gan
    else:
        n_samples_sim = min(n_samples, max_start_sim)
        n_samples_gan = min(n_samples, max_start_gan)

    rng = np.random.default_rng(seed)
    channels = ["T", "u", "v", "p"]

    if combine_channels:
        if C == 3:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        elif C == 4:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        else:
            ncols = min(C, 3)
            nrows = int(np.ceil(C / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axs = axs.flatten()

    sim_vars_all = []
    gan_vars_all = []

    for c in range(C):
        channel = channels[c]
        print(f"\nEvaluating temporal variance – Channel {channel}   {added_info}")

        # ==================================================
        # SIMULATION
        # ==================================================
        start_sim = rng.choice(max_start_sim, size=n_samples_sim, replace=False)
        sim_vars = []

        for start in start_sim:
            window = sim_tensors[c, start:start + N_window]
            v = compute_spatially_averaged_temporal_variance(
                window, max_lag
            )
            sim_vars.append(v)

        sim_vars = np.stack(sim_vars, axis=0)
        sim_mean, sim_lower, sim_upper = compute_temporal_mean_and_ci(sim_vars)
        sim_vars_all.append(sim_vars)

        # ==================================================
        # GAN (per seed)
        # ==================================================
        seed_means = []

        for s in range(S):
            start_gan = rng.choice(max_start_gan, size=n_samples_gan, replace=False)
            seed_vars = []

            for start in start_gan:
                window = gan_tensors[c, s, start:start + N_window]
                v = compute_spatially_averaged_temporal_variance(
                    window, max_lag
                )
                seed_vars.append(v)

            seed_vars = np.stack(seed_vars, axis=0)
            seed_mean, _, _ = compute_temporal_mean_and_ci(seed_vars)
            seed_means.append(seed_mean)

        seed_means = np.stack(seed_means, axis=0)
        gan_mean, gan_lower, gan_upper = compute_temporal_mean_and_ci(seed_means)
        gan_vars_all.append(seed_means)

        # ==================================================
        # Plotting (identical semantics to correlation)
        # ==================================================
        x = np.linspace(0, 1, max_lag)

        if combine_channels:
            ax = axs[c]
            ax.plot(x, sim_mean, label="Simulation", color="r")
            ax.plot(x, gan_mean, label="GAN", linestyle="dashed", color="b")
            ax.fill_between(x, sim_lower, sim_upper, color="r", alpha=0.2)
            if gan_ci:
                ax.fill_between(x, gan_lower, gan_upper, color="b", alpha=0.2)

            ax.set_title(f"Channel {channel}   {added_info}")
            ax.set_xlabel("Relative Time Lag $\\tau$")
            ax.set_ylabel("Variance")
            ax.legend()
            ax.grid(True)

        else:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fname = (
                    f"{save_dir}/{video_name}_temporal_variance_channel_{channel}_"
                    f"{added_info}.png"
                )
            else:
                fname = None

            plt.figure(figsize=(8, 4))
            plt.plot(x, sim_mean, label="Simulation", color="r")
            plt.plot(x, gan_mean, label="GAN", linestyle="dashed", color="b")
            plt.fill_between(x, sim_lower, sim_upper, color="r", alpha=0.2)
            if gan_ci:
                plt.fill_between(x, gan_lower, gan_upper, color="b", alpha=0.2)

            plt.xlabel("Relative Time Lag $\\tau$")
            plt.ylabel("Variance")
            plt.title(f"Temporal Variance – Channel {channel}   {added_info}")
            plt.legend()
            plt.grid(True)

            if fname:
                plt.savefig(fname)
                plt.close()
            else:
                plt.show()

    if combine_channels:
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = (
                f"{save_dir}/{video_name}_temporal_variance_all_channels_"
                f"{added_info}.png"
            )
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()

    return sim_vars_all, gan_vars_all
