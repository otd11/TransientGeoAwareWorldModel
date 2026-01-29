
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats

def compute_correlation_along_axis(tensors, direction='width', crop_h=120, crop_w=100, start_h=0, start_w=0):
    """
    tensors: 
      - (C, F, H, W) for sim_tensors
      - (C, S, F, H, W) for gan_tensors
    Returns:
      - For sim_tensors: list of C arrays, each (F, crop_dim)
      - For gan_tensors: list of C arrays, each (S, F, crop_dim)
    """
    if tensors.ndim == 5:
        num_channels, num_seeds, num_frames, H, W = tensors.shape
        all_correlations = [[[] for _ in range(num_seeds)] for _ in range(num_channels)]
        for seed in range(num_seeds):
            for frame in range(num_frames):
                for c in range(num_channels):
                    img = tensors[c, seed, frame, start_h:start_h + crop_h, start_w:start_w + crop_w]
                    corrs = []
                    if direction == 'width':
                        for px in range(crop_w):
                            corr = np.corrcoef(img[:, 0], img[:, px])[0, 1]
                            corrs.append(corr)
                    elif direction == 'height':
                        for px in range(crop_h):
                            corr = np.corrcoef(img[0, :], img[px, :])[0, 1]
                            corrs.append(corr)
                    else:
                        raise ValueError("direction must be 'width' or 'height'")
                    all_correlations[c][seed].append(corrs)
        return [np.array(ch_corrs) for ch_corrs in all_correlations]  # (S, F, crop_dim)

    else:
        num_channels, num_frames, H, W = tensors.shape
        all_correlations = [[] for _ in range(num_channels)]
        for frame in range(num_frames):
            for c in range(num_channels):
                img = tensors[c, frame, start_h:start_h + crop_h, start_w:start_w + crop_w]
                corrs = []
                if direction == 'width':
                    for px in range(crop_w):
                        corr = np.corrcoef(img[:, 0], img[:, px])[0, 1]
                        corrs.append(corr)
                elif direction == 'height':
                    for px in range(crop_h):
                        corr = np.corrcoef(img[0, :], img[px, :])[0, 1]
                        corrs.append(corr)
                else:
                    raise ValueError("direction must be 'width' or 'height'")
                all_correlations[c].append(corrs)
        return [np.array(corrs) for corrs in all_correlations]  # (F, crop_dim)

def compute_mean_and_ci(cor_matrix, alpha=0.05):
    """
    cor_matrix: (N, D) correlation matrix where N is number of samples
    """
    N, D = cor_matrix.shape
    if N <= 3:
        raise ValueError("Not enough samples for CI (need N > 3)")
    mean_corr = np.mean(cor_matrix, axis=0)
    mean_corr = np.clip(mean_corr, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + mean_corr) / (1 - mean_corr))
    std_z = 1 / np.sqrt(N - 3)
    z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
    z_upper = z + z_crit * std_z
    z_lower = z - z_crit * std_z
    upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    return mean_corr, lower, upper

def plot_correlation(mean_sim, mean_gan, sim_lower, sim_upper, lower, upper, channel, direction='width', save_path=None, gan_ci=False, added_info=""):
    D = mean_sim.shape[0]
    x = np.linspace(0, 1, D)
    plt.figure(figsize=(8, 4))
    plt.plot(x, mean_sim, label='Simulation', color='r')
    plt.plot(x, mean_gan, label='GAN', linestyle='dashed', color='b')
    plt.fill_between(x, sim_lower, sim_upper, color='r', alpha=0.2, label='Sim CI')

    if gan_ci:
        plt.fill_between(x, lower, upper, color='b', alpha=0.2, label='GAN CI')

    plt.title(f'{direction.capitalize()}-wise Correlation – Channel {channel}   ' + added_info)
    plt.xlabel('Relative Position $p$')
    plt.ylabel('Correlation $\\rho$')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_spatial_correlation(
    sim_tensors, gan_tensors, start_w=0, start_h=0, crop_w=100, crop_h=120,
    save_dir=None, added_info="", combine_channels=False, video_name=None, gan_ci=True
):
    sim_corrs = []
    gan_corrs = []
    channels = ["T", "u", "v", "p"]
    C = gan_tensors.shape[0]
    
    for direction in ['width', 'height']:
        print(f"\n----- {direction.upper()}-wise correlation -----")
        
        print("Computing simulation correlations...")
        sim_corr = compute_correlation_along_axis(sim_tensors, direction, crop_h, crop_w, start_h, start_w)  # (C) list of (F_sim, D)
        sim_corrs.append(sim_corr)
        
        print("Computing GAN correlations...")
        gan_corr = compute_correlation_along_axis(gan_tensors, direction, crop_h, crop_w, start_h, start_w)  # (C) list of (S, F_gan, D)
        gan_corrs.append(gan_corr)

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
            print(f"Channel {channel} " + added_info)

            # SIM: mean over frames → (F_sim, D)
            sim_mean, sim_lower, sim_upper = compute_mean_and_ci(sim_corr[c])

            # GAN: mean over frames first → (S, D), then compute mean/CI over seeds
            seed_means = np.mean(gan_corr[c], axis=1)  # (S, D)
            gan_mean, gan_lower, gan_upper = compute_mean_and_ci(seed_means)

            D = sim_mean.shape[0]
            x = np.linspace(0, 1, D)

            if combine_channels:
                ax = axs[c]
                ax.plot(x, sim_mean, label='Simulation', color='r')
                ax.plot(x, gan_mean, label='GAN', linestyle='dashed', color='b')
                ax.fill_between(x, sim_lower, sim_upper, color='r', alpha=0.2)

                if gan_ci:
                    ax.fill_between(x, gan_lower, gan_upper, color='b', alpha=0.2)
                    
                ax.set_title(f'{direction.capitalize()} – Channel {channel}   {added_info}')
                ax.set_xlabel('Relative Position $p$')
                ax.set_ylabel('Correlation $\\rho$')
                ax.legend()
                ax.grid(True)
            else:
                fname = None
                if save_dir:
                    fname = f"{save_dir}/{video_name}_cor_{direction}_c_{channel}_rec_{start_w}_{start_h}_{crop_w}_{crop_h}_" + added_info + ".png"
                plot_correlation(sim_mean, gan_mean, sim_lower, sim_upper, gan_lower, gan_upper, channel=channel, direction=direction, save_path=fname, added_info=added_info, gan_ci=gan_ci)

        if combine_channels:
            plt.tight_layout()
            if save_dir:
                fname = f"{save_dir}/{video_name}_cor_{direction}_c_{channel}_rec_{start_w}_{start_h}_{crop_w}_{crop_h}_{added_info}.png"
                plt.savefig(fname)
                plt.close()
            else:
                plt.show()

    return sim_corrs, gan_corrs
