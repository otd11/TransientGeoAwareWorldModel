import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm


def compute_temporal_autocorrelation_at_point(tensors, h, w, max_lag=None):
    """
    Compute per-channel temporal autocorrelation of the signal at (h, w) across time lags.
    tensors: np.ndarray of shape (C, N, H, W)
    Returns: np.ndarray of shape (C, max_lag)
    """
    if not isinstance(tensors, np.ndarray):
        tensors = tensors.cpu().numpy()

    C, N, H, W = tensors.shape
    if max_lag is None:
        max_lag = N

    corrs = []

    for c in range(C):
        ts = tensors[c, :, h, w]
        if np.all(np.isnan(ts)) or np.std(ts) == 0:
            corrs.append(np.full(max_lag, np.nan))
            continue

        ts = ts - np.nanmean(ts)
        ac = [np.corrcoef(ts[:-lag], ts[lag:])[0, 1] if lag > 0 else 1.0 for lag in range(max_lag)]
        corrs.append(ac)

    return np.array(corrs)


def compute_temporal_mean_and_ci(cor_matrix, alpha=0.05):
    """
    Compute mean and Fisher-transformed confidence interval across samples.
    cor_matrix: np.ndarray of shape (N, T)
    Returns: (mean, lower, upper), each of shape (T,)
    """
    cor_matrix = np.clip(cor_matrix, -0.9999, 0.9999)
    mean_corr = np.nanmean(cor_matrix, axis=0)

    N = cor_matrix.shape[0]
    z = 0.5 * np.log((1 + mean_corr) / (1 - mean_corr))
    std_z = 1 / np.sqrt(N - 3)
    z_crit = norm.ppf(1 - alpha / 2)

    z_upper = z + z_crit * std_z
    z_lower = z - z_crit * std_z

    upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)

    return mean_corr, lower, upper


def plot_temporal_correlation_single_point(
    mean_sim, mean_gan,
    sim_lower, sim_upper,
    gan_lower, gan_upper,
    channel, save_path=None, added_info="", gan_ci=False
):
    mean_sim = np.atleast_1d(mean_sim)
    mean_gan = np.atleast_1d(mean_gan)
    sim_lower = np.atleast_1d(sim_lower)
    sim_upper = np.atleast_1d(sim_upper)
    gan_lower = np.atleast_1d(gan_lower)
    gan_upper = np.atleast_1d(gan_upper)

    D = mean_sim.shape[0]
    x = np.linspace(0, 1, D)

    plt.figure(figsize=(8, 4))
    plt.plot(x, mean_sim, label='Simulation', color='r')
    plt.plot(x, mean_gan, label='GAN', linestyle='dashed', color='b')
    plt.fill_between(x, sim_lower, sim_upper, color='r', alpha=0.2, label='Sim CI')

    if gan_ci:
        plt.fill_between(x, gan_lower, gan_upper, color='b', alpha=0.2, label='GAN CI')

    plt.title(f'Temporal Correlation – Channel {channel}   {added_info}')
    plt.xlabel('Relative Time Lag $\\tau$')
    plt.ylabel('Correlation $\\rho$')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_temporal_correlation_absolute_frames(
    mean_sim, mean_gan,
    sim_lower, sim_upper,
    gan_lower, gan_upper,
    channel,
    T_sim,
    T_gan,
    save_path=None,
    added_info="",
    gan_ci=False
):
    """
    Plot temporal correlation against absolute frame numbers.
    Simulation: correlation over T_sim frames
    GAN: correlation over first T_gan frames
    """
    x_sim = np.arange(T_sim)
    x_gan = np.arange(T_gan)

    plt.figure(figsize=(8, 4))
    plt.plot(x_sim, mean_sim[:T_sim], label='Simulation', color='r')
    plt.plot(x_gan, mean_gan[:T_gan], label='GAN', linestyle='dashed', color='b')
    plt.fill_between(x_sim, sim_lower[:T_sim], sim_upper[:T_sim], color='r', alpha=0.2, label='Sim CI')

    if gan_ci:
        plt.fill_between(x_gan, gan_lower[:T_gan], gan_upper[:T_gan], color='b', alpha=0.2, label='GAN CI')

    plt.title(f'Temporal Correlation vs. Frame Number – Channel {channel}   {added_info}')
    plt.xlabel('Absolute Frame Number')
    plt.ylabel('Correlation $\\rho$')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_temporal_correlation(
    sim_tensors,
    gan_tensors,
    point,
    N_window,
    save_dir=None,
    added_info="",
    max_lag=None,
    n_samples=100,
    seed=42,
    all_windows=False,
    combine_channels=False,
    video_name=None,
    gan_ci=True
):
    """
    Compare temporal autocorrelation at a single spatial point (h, w) using overlapping windows of size N_window.
    - sim_tensors: shape (C, F_sim, H, W)
    - gan_tensors: shape (C, S, F_gan, H, W)
    - N_window: number of frames per window for both sim and gan data
    """
    w, h = point
    _, F_sim, H, W = sim_tensors.shape
    C, S, F_gan, _, _ = gan_tensors.shape

    if max_lag is None:
        max_lag = N_window

    # --- SIMULATION WINDOWING SETUP ---
    max_start_sim = F_sim - N_window + 1
    if max_start_sim < 1:
        raise ValueError("sim_tensors is too short compared to N_window.")

    # --- GAN WINDOWING SETUP (per seed) ---
    max_start_gan = F_gan - N_window + 1
    if max_start_gan < 1:
        raise ValueError("gan_tensors is too short compared to N_window.")

    if all_windows:
        n_samples_sim = max_start_sim
        n_samples_gan = max_start_gan
    else:
        n_samples_sim = min(n_samples, max_start_sim)
        n_samples_gan = min(n_samples, max_start_gan)

    rng = np.random.default_rng(seed)
    channels = ["T", "u", "v", "p"]

    sim_corr = []
    gan_corr = []

    if combine_channels:
        # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        # axs = axs.flatten()
        if C == 3:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs = axs.flatten()
        elif C == 4:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
        else:
            # Generic fallback for any other number of channels
            ncols = min(C, 3)
            nrows = int(np.ceil(C / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axs = axs.flatten()

    for c in range(C):
        channel = channels[c]
        print(f"\nEvaluating Channel {channel} at point (h={h}, w={w}) {added_info}")

        # ==================================================
        # SIMULATION: Overlapping windows
        # ==================================================
        start_indices_sim = rng.choice(max_start_sim, size=n_samples_sim, replace=False)
        sim_acfs = []
        for start in start_indices_sim:
            ts = sim_tensors[c, start:start + N_window, h, w]
            ts = ts - np.nanmean(ts)
            if np.nanstd(ts) == 0:
                acf = np.full(max_lag, np.nan)
            else:
                acf = [np.corrcoef(ts[:-lag], ts[lag:])[0, 1] if lag > 0 else 1.0 for lag in range(max_lag)]
            sim_acfs.append(acf)
        sim_acfs = np.stack(sim_acfs, axis=0)
        sim_mean, sim_lower, sim_upper = compute_temporal_mean_and_ci(sim_acfs)
        sim_corr.append(sim_acfs)

        # ==================================================
        # GAN: Per-seed windowed correlations
        # ==================================================
        seed_acfs_all = []
        for s in range(S):
            start_indices_gan = rng.choice(max_start_gan, size=n_samples_gan, replace=False)
            gan_acfs_seed = []
            for start in start_indices_gan:
                ts = gan_tensors[c, s, start:start + N_window, h, w]
                ts = ts - np.nanmean(ts)
                if np.nanstd(ts) == 0:
                    acf = np.full(max_lag, np.nan)
                else:
                    acf = [np.corrcoef(ts[:-lag], ts[lag:])[0, 1] if lag > 0 else 1.0 for lag in range(max_lag)]
                gan_acfs_seed.append(acf)
            gan_acfs_seed = np.stack(gan_acfs_seed, axis=0)

            # CI per seed (optional)
            seed_mean, _, _ = compute_temporal_mean_and_ci(gan_acfs_seed)
            seed_acfs_all.append(seed_mean)

        seed_acfs_all = np.stack(seed_acfs_all, axis=0)
        gan_mean, gan_lower, gan_upper = compute_temporal_mean_and_ci(seed_acfs_all)
        gan_corr.append(seed_acfs_all)

        # ==================================================
        # PLOTTING
        # ==================================================
        D = sim_mean.shape[0]
        x = np.linspace(0, 1, D)

        if combine_channels:
            ax = axs[c]
            ax.plot(x, sim_mean, label='Simulation', color='r')
            ax.plot(x, gan_mean, label='GAN', linestyle='dashed', color='b')
            ax.fill_between(x, sim_lower, sim_upper, color='r', alpha=0.2)
            if gan_ci:
                ax.fill_between(x, gan_lower, gan_upper, color='b', alpha=0.2)

            ax.set_title(f'Channel {channel}   {added_info}')
            ax.set_xlabel('Relative Time Lag $\\tau$')
            ax.set_ylabel('Correlation $\\rho$')
            ax.legend()
            ax.grid(True)
        else:
            fname = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{save_dir}/{video_name}_temporal_corr_channel_{channel}_point_{h}_{w}_" + added_info + ".png"
            plot_temporal_correlation_single_point(
                sim_mean, gan_mean, sim_lower, sim_upper,
                gan_lower, gan_upper,
                channel=channel,
                save_path=fname,
                added_info=added_info,
                gan_ci=gan_ci
            )

    if combine_channels:
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{video_name}_temporal_corr_all_channels_point_{h}_{w}_" + added_info + ".png"
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()

    return sim_corr, gan_corr
