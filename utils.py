# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import contextlib
import functools
import math
import os
import pickle
import re
import time
from collections.abc import Iterable, Iterator
from typing import Optional, Union

import einops
import imageio
import PIL.Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np

import dnnlib
import torch_utils.distributed as dist_utils
from torch_utils import distributed


#####


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import io


# =====================================================================================================================


def get_next_run_dir(outdir: str, desc: Optional[str] = None) -> str:
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    name = f"{cur_run_id:05d}" if desc is None else f"{cur_run_id:05d}-{desc}"
    run_dir = os.path.join(outdir, name)
    assert not os.path.exists(run_dir)
    return run_dir


# =====================================================================================================================


def load_G(path: str):
    with dnnlib.util.open_url(path) as fp:
        G = pickle.load(fp)

        if "-train.pkl" in path:
            print("keys in G:", G.keys())
            G = G["G"]  # For training checkpoints, use the ema model.

    return G.requires_grad_(False).eval().cuda()


# =====================================================================================================================


def rank0_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist_utils.get_rank() == 0:
            return func(*args, **kwargs)

    return wrapper


@rank0_only
def print0(*args, **kwargs):
    print(*args, **kwargs)


@contextlib.contextmanager
def context_timer0(message: str):
    start_time = time.time()
    print0(message, end="... ")
    try:
        yield
    finally:
        duration = time.time() - start_time
        print0(f"{duration:.2f}s")


# =====================================================================================================================


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist_utils.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor = torch.cat(tensor_list)
    return tensor


def all_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, dist.ReduceOp.SUM)
    tensor = tensor / dist_utils.get_world_size()
    return tensor


def sharded_all_mean(tensor: torch.Tensor, shard_size: int = 2**23) -> torch.Tensor:
    assert tensor.dim() == 1
    shards = tensor.tensor_split(math.ceil(tensor.numel() / shard_size))
    for shard in shards:
        torch.distributed.all_reduce(shard)
    tensor = torch.cat(shards) / dist_utils.get_world_size()
    return tensor


# =====================================================================================================================


def sync_grads(network: nn.Module, gain: Optional[torch.Tensor] = None):
    params = [param for param in network.parameters() if param.grad is not None]
    flat_grads = torch.cat([param.grad.flatten() for param in params])
    flat_grads = sharded_all_mean(flat_grads)
    flat_grads = flat_grads if gain is None else flat_grads * gain
    torch.nan_to_num(flat_grads, nan=0, posinf=1e5, neginf=-1e5, out=flat_grads)
    grads = flat_grads.split([param.numel() for param in params])
    for param, grad in zip(params, grads):
        param.grad = grad.reshape(param.size())


# =====================================================================================================================


def random_seed(max_seed: int = 2**31 - 1) -> int:
    seed = torch.randint(max_seed + 1, (), device="cuda")
    if distributed.get_world_size() > 1:
        dist.broadcast(seed, src=0)
    return seed.item()


# =====================================================================================================================


def multiple_nearest_sqrt(number: int) -> int:
    for i in range(int(math.sqrt(number)), 0, -1):
        if number % i == 0:
            return i


def write_video_grid(
    segments: Union[torch.Tensor, Iterable[torch.Tensor]],
    path: Optional[os.PathLike] = None,
    fps: int = 30,
    max_samples: Optional[int] = None,
    num_rows: Optional[int] = None,
    to_uint8: bool = True,
    gather: bool = False,
):
    if isinstance(segments, torch.Tensor):
        segments = [segments]




    if dist_utils.get_rank() == 0:

        # Convert the path to a string and split at "/samples/"
        path_str = str(path)
        split_index = path_str.find("/samples/")

        if split_index != -1:  # If "/samples/" exists in the path
            # Split the path into two parts and insert the new string after "/samples/"
            new_path_str = path_str[:split_index + len("/samples/")] + "RGB_" + path_str[split_index + len("/samples/"):]
            video_path = new_path_str 

        assert path is not None
        video_writer = imageio.get_writer(video_path, mode="I", fps=fps, codec="libx264", bitrate="16M")

    for segment in segments:

        segment = (segment * 127.5 + 128).clamp(0, 255).to(torch.uint8) if to_uint8 else segment
        segment = all_gather(segment) if gather else segment


        if dist_utils.get_rank() == 0:
            segment = segment[:max_samples] if max_samples else segment
            num_rows = num_rows or multiple_nearest_sqrt(segment.size(0))

            


            for frame in segment.unbind(dim=2):
                frame_grid = einops.rearrange(frame, "(nw nh) c h w -> (nh h) (nw w) c", nh=num_rows)

                # Ensures each edge is a multiple of 16, resizing if needed.
                scale_y = 16 // math.gcd(frame_grid.size(0), 16)
                scale_x = 16 // math.gcd(frame_grid.size(1), 16)
                scale = scale_y * scale_x // math.gcd(scale_y, scale_x)
                if scale > 1:
                    frame_grid = einops.rearrange(frame_grid, "h w c -> 1 c h w")
                    frame_grid = F.interpolate(frame_grid, scale_factor=scale, mode="nearest")
                    frame_grid = einops.rearrange(frame_grid, "1 c h w -> h w c")


                frame_grid = frame_grid.cpu().numpy()
                # Flip the frame vertically
                frame_grid = np.flipud(frame_grid)


                # Convert to PIL image and resize to frame_size
                img = PIL.Image.fromarray((frame_grid * 255).astype(np.uint8))  # Convert to 8-bit
                img_resized = img.resize((640, 640), PIL.Image.Resampling.LANCZOS)

                # Convert back to ndarray
                img_resized = np.array(img_resized)

                video_writer.append_data(img_resized)

                

    if dist_utils.get_rank() == 0:
        video_writer.close()


# =====================================================================================================================


def save_image_grid(
    image: torch.Tensor,
    path: Optional[os.PathLike] = None,
    max_samples: Optional[int] = None,
    num_rows: Optional[int] = None,
    to_uint8: bool = True,
    gather: bool = False,
):
    if dist_utils.get_rank() == 0:
        assert path is not None

    image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8) if to_uint8 else image
    image = all_gather(image) if gather else image

    if dist_utils.get_rank() == 0:
        image = image[:max_samples] if max_samples else image
        num_rows = num_rows or multiple_nearest_sqrt(image.size(0))
        image_grid = einops.rearrange(image, "(nw nh) c h w -> (nh h) (nw w) c", nh=num_rows)
        PIL.Image.fromarray(image_grid.cpu().numpy()).save(path)


# ====================================================================================================================


def get_infinite_data_iter(dataset: Dataset, seed: Optional[int] = None, **loader_kwargs) -> Iterator:
    seed = random_seed() if seed is None else seed
    generator = torch.Generator().manual_seed(seed)
    sampler = DistributedSampler(dataset, seed=seed) if distributed.get_world_size() > 1 else None
    loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler, generator=generator, **loader_kwargs)

    epoch = 0
    while True:
        if distributed.get_world_size() > 1:
            sampler.set_epoch(epoch)
        for sample in loader:
            yield sample
        epoch += 1


# =====================================================================================================================




def write_video_for_each_channel(
    segments: Union[torch.Tensor, Iterable[torch.Tensor]],
    path_prefix: Optional[os.PathLike] = None,
    fps: int = 30,
    max_samples: Optional[int] = None,
    num_rows: Optional[int] = None,
    plot_mask: bool = True,
    gather: bool = False,
    onlyT: bool = False,
    channels: int = 3, 
    cond: Optional[dict[torch.Tensor]] = None,
    legend: bool = True,
):
    





    if isinstance(segments, torch.Tensor):
        segments = [segments]

    if dist_utils.get_rank() == 0:
        assert path_prefix is not None, "You must specify a path prefix for saving videos."

    if cond is not None:

        cond_sp = cond["cond_sp"].cpu()
        
        if cond["cond_num"] is not None and len(cond["cond_num"][0]) > 1:
            cond_num = cond["cond_num"].cpu().numpy()
            cond_num = cond_num[0]  # we only need the first batch
            
            T_amb = round((cond_num[0] * 15) + 299, 1)
            Q_a = round(cond_num[1] * 100, 1)
            Q_b = round(cond_num[2] * 100, 1)
        else:
            T_amb = 300
            Q_a = 100
            Q_b = 0
        
        if cond_sp.ndim == 3:
            cond_sp = cond_sp.unsqueeze(0)  # Add batch dim 
            

        

        cond_sliced = cond_sp[0, :, :, :] # we only need the first batch
        

        
        
        cond_sliced = cond_sliced.numpy()
        

        # Plot cond[0] with transparency
        cond_channel_0 = cond_sliced[0]

        # Create a mask for plotting
        if np.array_equal(cond_channel_0, cond_channel_0.astype(bool)):
            cond_channel_0_masked_tmp = np.where(cond_channel_0 == 1, cond_channel_0, np.nan)
        else:       
            cond_channel_0_masked_tmp = np.where(cond_channel_0 >= 0, 1, np.nan)
            # Set boundary values to 0
            cond_channel_0_masked_tmp[0, :] = np.nan
            cond_channel_0_masked_tmp[-1, :] = np.nan
            cond_channel_0_masked_tmp[:, 0] = np.nan
            cond_channel_0_masked_tmp[:, -1] = np.nan
            
        cond_channel_0_masked = np.flipud(cond_channel_0_masked_tmp)
        



    video_writers = {}  # Dictionary to store video writers for each channel


    channel_names = ["T", "u", "v", "p"]   

    # [T_min, T_max]:  [299, 314]
    # [u_min, u_max]:  [-0.2, 0.2]
    # [v_min, v_max]:  [-0.16, 0.24]
    # [p_min, p_max]:  [-6.3, 6.7]
    vmins = [299, -0.2, -0.16, -6.3]  # Minimum values for each channel
    vmaxs = [314, 0.2, 0.24, 6.7]  # Maximum values for each channel

    channel_names = channel_names[:channels]
    vmins = vmins[:channels]
    vmaxs = vmaxs[:channels]
    
    if onlyT:
        channel_names = ["T"]

    for channel in range(len(channel_names)):  # Iterate over channels
        tmp_name = channel_names[channel]
        
        if dist_utils.get_rank() == 0:
            tmp_str = tmp_name + "_"

            # Convert the path to a string
            path_str = str(path_prefix)

            # Find the last occurrence of "/"
            last_slash_index = path_str.rfind("/")
            if last_slash_index != -1:  # If "/" exists in the path
                # Insert the new string after the last "/"
                new_path_str = path_str[:last_slash_index + 1] + tmp_str + path_str[last_slash_index + 1:]
                video_path = new_path_str
            else:
                video_path = f"{path_prefix}_channel_{tmp_name}.mp4"


            video_writer = imageio.get_writer(video_path, mode="I", fps=fps, codec="libx264", bitrate="16M")
            video_writers[tmp_name] = video_writer  # Store it in the dictionary



    for segment in segments:

        segment = all_gather(segment) if gather else segment
        

        if dist_utils.get_rank() == 0:
            segment = segment[:max_samples] if max_samples else segment



          
            ###########################################################################
            ## Normalization of the dataset

            # ########### Normalize to [0, 1]

            # [T_min, T_max]:  [299, 314]
            # [u_min, u_max]:  [-0.2, 0.2]
            # [v_min, v_max]:  [-0.16, 0.24]
            # [p_min, p_max]:  [-6.3, 6.7]


            
            # Inverse transformation for the first channel -> [T_min, T_max]
            segment[:, 0, :, :, :] = (segment[:, 0, :, :, :] * 15) + 299

            # Inverse transformation for the second channel -> [u_min, u_max]
            segment[:, 1, :, :, :] = (segment[:, 1, :, :, :] * 0.4) - 0.2

            # Inverse transformation for the third channel -> [v_min, v_max]
            segment[:, 2, :, :, :] = (segment[:, 2, :, :, :] * 0.4) - 0.16
         
            if segment.size(1) > 3:  # Check if the number of channels is greater than 3
                # Inverse transformation for the third channel -> [p_min, p_max]
                segment[:, 3, :, :, :] = (segment[:, 3, :, :, :] * 13) - 6.3  






            for channel in range(len(channel_names)):  
                tmp_name = channel_names[channel]

                channel_frames = segment[:, channel]  # Extract channel data
                vmin = vmins[channel]  # Get the minimum value for the current channel
                vmax = vmaxs[channel]  # Get the maximum value for the current channel
                
                for frame_tmp in channel_frames.unbind(dim=1):  # Iterate through time dimension (T)

                    # just plot for the first batch
                    frame = frame_tmp[0]                    
                    frame_np = frame.squeeze().cpu().numpy()

                    # Flip the array if you want origin='lower'
                    frame_np_flipped = np.flipud(frame_np)

                    # Convert to an image and write to video
                    fig, ax = plt.subplots(figsize=(6, 6))  # Create a single plot with fixed size

                    # Display the image
                    im = ax.imshow(frame_np_flipped, aspect="auto", cmap="coolwarm") #, vmin=vmin, vmax=vmax)

                    # Overlay the mask if needed
                    if cond is not None and plot_mask:
                        ax.imshow(cond_channel_0_masked, aspect="auto", cmap="gray", alpha=0.5)

                    # Hide axes
                    ax.axis("off")

                    if legend:
                        # Add the legend in the lower-right corner with transparency
                        legend_text = f"T_amb={T_amb}, Q_a={Q_a}, Q_b={Q_b}"
                        ax.text(
                            0.95, 0.05, legend_text,
                            ha='right', va='bottom', fontsize=12,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                            transform=ax.transAxes
                        )
                    

                    # Render the plot to a buffer (image)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                    buf.seek(0)



                    # Read the image from the buffer
                    frame_image = imageio.imread(buf)

                    # Close the buffer and figure to release memory
                    buf.close()
                    plt.close(fig)

                    # Ensure the frame has a consistent size (optional: resize image to fixed dimensions)
                    desired_width = 640
                    desired_height = 640

                    # Resize the frame to ensure consistent size across all frames
                    frame_image_resized = np.array(PIL.Image.fromarray(frame_image).resize((desired_width, desired_height)))
                

                    # Append the resized image to the video writer
                    video_writers[tmp_name].append_data(frame_image_resized)
            
    
    if dist_utils.get_rank() == 0:
        for channel in range(len(channel_names)):  # Iterate over channels
            tmp_name = channel_names[channel]
            video_writers[tmp_name].close()


