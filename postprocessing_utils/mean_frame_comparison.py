
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import imageio




def plot_channel_means(
    sim_tensors,
    gan_tensors,
    channel_names=["T", "u", "v", "p"],
    save_path=None,
    max_legend_seeds=8,
    experiment_name: str = "",
):
    """
    Optimized + saves each channel plot individually.

    Args:
        sim_tensors (torch.Tensor): shape (C, N_sim, H, W)
        gan_tensors (torch.Tensor): shape (C, Seeds, N_gan, H, W)
        channel_names (list[str])
        save_path (str): directory or file prefix
        experiment_name (str)
    """

    # -----------------------------
    # Extract shapes
    # -----------------------------
    _, N_sim, H, W = sim_tensors.shape
    C, Seeds, N_gan, _, _ = gan_tensors.shape

    # -----------------------------
    # Move everything to CPU NumPy once
    # -----------------------------
    sim_np = sim_tensors.mean(dim=(2, 3)).cpu().numpy()          # (C, N_sim)
    gan_np = gan_tensors.mean(dim=(3, 4)).cpu().numpy()          # (C, Seeds, N_gan)
    gan_mean_np = gan_np.mean(axis=1)                            # (C, N_gan)

    x_sim = np.arange(N_sim)
    x_gan = np.arange(N_gan)

    # -----------------------------
    # Precompute shifts per seed based on channel 0
    # -----------------------------
    sim_c0 = sim_np[0]
    gan_c0 = gan_np[0]

    # shift for GAN mean based on first GAN frame
    idx_global_shift = np.argmin(np.abs(sim_c0 - gan_mean_np[0, 0]))
    global_shift = x_sim[idx_global_shift]
    x_gan_mean_shifted = x_gan + global_shift

    # seed-wise shifts (only need for channels where we plot seeds)
    max_plot_seeds = min(Seeds, 20)
    seed_shifts = np.zeros(max_plot_seeds, dtype=int)

    for s in range(max_plot_seeds):
        y0 = gan_c0[s, 0]
        idx = np.argmin(np.abs(sim_c0 - y0))
        seed_shifts[s] = x_sim[idx]

    # -----------------------------
    # Prepare save directory
    # -----------------------------
    if save_path:
        if os.path.isdir(save_path):
            save_dir = save_path
        else:
            save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    exp_suffix = f"_{experiment_name}" if experiment_name else ""

    # -----------------------------
    # Color maps
    # -----------------------------
    if Seeds <= 10:
        cmap = plt.get_cmap("tab10")
    elif Seeds <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("hsv")

    colors = [cmap(i / max_plot_seeds) for i in range(max_plot_seeds)]

    sim_color = "#1f77b4"
    mean_color = "#ff7f0e"

    # -----------------------------
    # Plot each channel individually
    # -----------------------------
    for c in range(C):
        fig, ax = plt.subplots(figsize=(7, 5))

        # Simulation curve
        ax.plot(x_sim, sim_np[c], color=sim_color, linewidth=2.5,
                label=f"Sim {channel_names[c]}")

        # ----------- Plot raw seeds for specific channels -----------
        plot_seeds = (c == 0) or (c == 3)  # match your original rule

        if plot_seeds:
            for s in range(max_plot_seeds):
                shift = seed_shifts[s]
                ax.plot(
                    x_gan + shift,
                    gan_np[c, s],
                    "--",
                    linewidth=1.0,
                    alpha=0.5,
                    color=colors[s],
                    label=None if s >= max_legend_seeds else f"GAN seed {s}",
                )

            # GAN mean with shift
            ax.plot(
                x_gan_mean_shifted,
                gan_mean_np[c],
                linestyle="--",
                color=mean_color,
                linewidth=2.5,
                label=f"GAN mean {channel_names[c]}",
            )
        else:
            # Only GAN mean, without shift
            ax.plot(
                x_gan,
                gan_mean_np[c],
                linestyle="--",
                color=mean_color,
                linewidth=2.5,
                label=f"GAN mean {channel_names[c]}",
            )

        # Formatting
        ax.set_title(f"{channel_names[c]}")
        ax.set_xlabel("Frame index")
        ax.set_ylabel(f"{channel_names[c]} Mean value")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(ncol=2, fontsize=9, frameon=False)

        plt.tight_layout()

        # Save per-channel figure
        if save_dir:
            fname = f"channel_{c}_{channel_names[c]}{exp_suffix}.png"
            out_path = os.path.join(save_dir, fname)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_path}")

        plt.show()
        plt.close()




def find_closest_sim_frame(sim_tensors, gan_tensors, seed):
    """
    Find the simulation index closest to GAN frame mean for channel 0.
    Returns a single index for all channels.
    """
    # Convert to numpy
    if isinstance(sim_tensors, torch.Tensor):
        sim_np = sim_tensors.detach().cpu().numpy()
    else:
        sim_np = sim_tensors

    if isinstance(gan_tensors, torch.Tensor):
        gan_np = gan_tensors.detach().cpu().numpy()
    else:
        gan_np = gan_tensors

    # Use only channel 0
    gan_frame = gan_np[0, seed, 0]
    gan_mean = gan_frame.mean()
    sim_means = sim_np[0].reshape(sim_np.shape[1], -1).mean(axis=1)
    closest_idx = np.argmin(np.abs(sim_means - gan_mean))

    print(f"Closest simulation frame index (channel 0) = {closest_idx}")

    return closest_idx


def plot_comparison(sim_tensors, gan_tensors, index, seed,
                    save_path=None, experiment_name="exp", save=True):
    """
    For all channels:
      - Channel 0: use frame at `index`.
      - Other channels: average frames around `index` ([index-2:index+2]).
      - Hardcoded channel names, vmin, vmax.
      - Optionally save plots.
    """
    CHANNEL_NAMES = ["T", "u", "v", "p"]
    VMIN = [299, -0.2, -0.16, -6.3]
    VMAX = [314, 0.2, 0.24, 6.7]

    # Convert to numpy
    if isinstance(sim_tensors, torch.Tensor):
        sim_np = sim_tensors.detach().cpu().numpy()
    else:
        sim_np = sim_tensors

    if isinstance(gan_tensors, torch.Tensor):
        gan_np = gan_tensors.detach().cpu().numpy()
    else:
        gan_np = gan_tensors

    c, n_sim, h, w = sim_np.shape

    if save_path is not None and save:
        os.makedirs(save_path, exist_ok=True)

    for ch in range(c):
        gan_frame = gan_np[ch, seed, 0]

        # Use same index for all channels
        if ch == 0:
            sim_frame = sim_np[ch, index]
        else:
            start = max(0, index - 2)
            end = min(n_sim, index + 3)
            sim_frame = np.mean(sim_np[ch, start:end], axis=0)

        # Flip vertically
        gan_frame_flipped = np.flipud(gan_frame)
        sim_frame_flipped = np.flipud(sim_frame)

        # Side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        im1 = axes[0].imshow(gan_frame_flipped, aspect="auto", cmap="coolwarm",
                             vmin=VMIN[ch], vmax=VMAX[ch])
        axes[0].set_title(f"GAN ({CHANNEL_NAMES[ch]}), frame=0")
        axes[0].axis("off")

        im2 = axes[1].imshow(sim_frame_flipped, aspect="auto", cmap="coolwarm",
                             vmin=VMIN[ch], vmax=VMAX[ch])
        if ch == 0:
            axes[1].set_title(f"Sim ({CHANNEL_NAMES[ch]}), frame={index}")
        else:
            axes[1].set_title(f"Sim ({CHANNEL_NAMES[ch]}), avg frame=[{max(0,index-2)}:{min(n_sim,index+3)}]")
        axes[1].axis("off")

        fig.colorbar(im2, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04)

        # Save if requested
        if save and save_path is not None:
            filename = f"{experiment_name}_ch{ch}_s{seed}_{CHANNEL_NAMES[ch]}.png"
            full_path = os.path.join(save_path, filename)
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot for channel {ch} to: {full_path}")

        plt.show()
        
        



def create_comparison_videos(sim_tensors, gan_tensors, index, seed,
                             save_path=None, experiment_name="exp", save=True,
                             fps=30, caption=None, caption_color="black"):
    """
    Create comparison videos for each channel.
    For each GAN frame k:
        - Compare GAN[ch, seed, k] with SIM[ch, index + k].
    Saves one MP4 video per channel.

    caption: Optional string shown on the whole frame
    caption_color: Color of the caption text
    """

    CHANNEL_NAMES = ["T", "u", "v", "p"]
    VMIN = [299, -0.2, -0.16, -6.3]
    VMAX = [314, 0.2, 0.24, 6.7]

    # Convert to numpy
    if isinstance(sim_tensors, torch.Tensor):
        sim_np = sim_tensors.detach().cpu().numpy()
    else:
        sim_np = sim_tensors

    if isinstance(gan_tensors, torch.Tensor):
        gan_np = gan_tensors.detach().cpu().numpy()
    else:
        gan_np = gan_tensors

    c, n_sim, h, w = sim_np.shape
    _, _, n_gan, _, _ = gan_np.shape  # channels, seeds, frames, H, W

    # Prepare save path
    if save_path is not None and save:
        os.makedirs(save_path, exist_ok=True)

    # Loop through channels
    for ch in range(c):
        frames = []

        for k in range(n_gan):
            sim_idx = index + k
            if sim_idx >= n_sim:
                print(f"Simulation index {sim_idx} out of range. Stopping channel {ch}.")
                break

            # Extract frames
            gan_frame = gan_np[ch, seed, k]
            sim_frame = sim_np[ch, sim_idx]

            # Flip vertically
            gan_frame_flipped = np.flipud(gan_frame)
            sim_frame_flipped = np.flipud(sim_frame)

            # Plot comparison frame
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(
                gan_frame_flipped,
                aspect="auto",
                cmap="coolwarm",
                vmin=VMIN[ch], vmax=VMAX[ch],
            )
            axes[0].set_title(f"GAN ({CHANNEL_NAMES[ch]})")
            axes[0].axis("off")

            axes[1].imshow(
                sim_frame_flipped,
                aspect="auto",
                cmap="coolwarm",
                vmin=VMIN[ch], vmax=VMAX[ch],
            )
            axes[1].set_title(f"Sim ({CHANNEL_NAMES[ch]})")
            axes[1].axis("off")

            # ---- GLOBAL CAPTION ----
            if caption is not None:
                fig.text(
                    0.5, 0.98, caption,
                    ha="center", va="top",
                    fontsize=14,
                    color=caption_color,
                    weight="bold"
                )

            # Convert image to array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frames.append(img)
            plt.close(fig)

        # Save the video
        if save and save_path is not None:
            video_filename = f"{experiment_name}_ch{ch}_s{seed}_{CHANNEL_NAMES[ch]}.mp4"
            video_path = os.path.join(save_path, video_filename)
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"Saved video for channel {ch} to: {video_path}")


def create_combined_comparison_video(
    sim_tensors,
    gan_tensors,
    index,
    seed,
    save_path=None,
    experiment_name="exp",
    save=True,
    fps=30,
    caption=None,
    caption_color="black",
):
    """
    Create ONE combined comparison video as a 2x2 grid.

    Layout:
        Row 1: channels 0 and 3
        Row 2: channels 1 and 2

    Each cell:
        [ GAN | separator | SIM ]
    """

    CHANNEL_NAMES = ["T", "u", "v", "p"]
    VMIN = [299, -0.2, -0.16, -6.3]
    VMAX = [314, 0.2, 0.24, 6.7]

    SEPARATOR_WIDTH = 6
    SEPARATOR_VALUE = 0.0  # neutral value for coolwarm

    # Convert to numpy
    sim_np = sim_tensors.detach().cpu().numpy() if isinstance(sim_tensors, torch.Tensor) else sim_tensors
    gan_np = gan_tensors.detach().cpu().numpy() if isinstance(gan_tensors, torch.Tensor) else gan_tensors

    c, n_sim, h, w = sim_np.shape
    _, _, n_gan, _, _ = gan_np.shape

    # Channel order for 2x2 layout
    channel_order = [0, 3, 1, 2]

    # Prepare save path
    if save and save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    frames = []

    for k in range(n_gan):
        sim_idx = index + k
        if sim_idx >= n_sim:
            print(f"Simulation index {sim_idx} out of range. Stopping.")
            break

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for ax, ch in zip(axes.flat, channel_order):
            gan_frame = np.flipud(gan_np[ch, seed, k])
            sim_frame = np.flipud(sim_np[ch, sim_idx])

            # Separator between GAN and SIM
            separator = np.full(
                (gan_frame.shape[0], SEPARATOR_WIDTH),
                SEPARATOR_VALUE,
                dtype=gan_frame.dtype,
            )

            combined = np.concatenate(
                [gan_frame, separator, sim_frame],
                axis=1,
            )

            ax.imshow(
                combined,
                cmap="coolwarm",
                vmin=VMIN[ch],
                vmax=VMAX[ch],
                aspect="auto",
            )

            ax.set_title(f"{CHANNEL_NAMES[ch]}  (GAN | SIM)")
            ax.axis("off")

        # Global caption
        if caption is not None:
            fig.text(
                0.5,
                0.98,
                caption,
                ha="center",
                va="top",
                fontsize=14,
                color=caption_color,
                weight="bold",
            )

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Convert figure to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)

        plt.close(fig)

    # Save video
    if save and save_path is not None:
        video_path = os.path.join(
            save_path, f"{experiment_name}_combined_s{seed}.mp4"
        )
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Saved combined video to: {video_path}")




def plot_sim_mean_std(sim_tensors, 
                      channel_names=("T", "u", "v", "p"),
                      save_path=None,
                      save=True):
    """
    For each channel in sim_tensors:
        - Compute mean over all pixels for each frame.
        - Compute std  over all pixels for each frame.
    Then plot mean ± std as a 1D time-series curve.

    Args:
        sim_tensors: (C, N_frames, H, W)
        channel_names: tuple/list of channel names for labeling
        save_path: directory to save PNG plots (optional)
        save: whether to save the plots
    """

    # Convert PyTorch → NumPy if needed
    if isinstance(sim_tensors, torch.Tensor):
        sim_np = sim_tensors.detach().cpu().numpy()
    else:
        sim_np = sim_tensors

    # Expected shape: (channels, n_frames, H, W)
    c, n_frames, h, w = sim_np.shape
    frames = np.arange(n_frames)

    # Create directory if saving
    if save and save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    for ch in range(c):
        # Extract channel: shape (n_frames, H, W)
        data = sim_np[ch]

        # Flatten each frame → shape: (n_frames, H*W)
        flat = data.reshape(n_frames, -1)

        # Compute mean/std per frame
        means = flat.mean(axis=1)
        stds  = flat.std(axis=1)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(frames, means, label=f"{channel_names[ch]} mean", linewidth=2)

        # Shade mean ± std region
        plt.fill_between(
            frames,
            means - stds,
            means + stds,
            alpha=0.3,
            label=f"{channel_names[ch]} ± std"
        )

        plt.title(f"Simulation Mean ± Std — Channel: {channel_names[ch]}")
        plt.xlabel("Frame Index")
        plt.ylabel(f"{channel_names[ch]}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save if requested
        if save and save_path is not None:
            filename = f"{channel_names[ch]}_mean_std.png"
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {full_path}")

        plt.show()
