# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pathlib

import click
import torch
import time

import utils

import numpy as np

# =====================================================================================================================
def create_cond_sp(n_w, n_h, x_min, x_max, y_min, y_max, circle1_x, circle1_y, circle1_radius, circle2_x, circle2_y, circle2_radius):
   # Define the unit square domain
    x = np.linspace(x_min, x_max, n_w)
    y = np.linspace(y_min, y_max, n_h)
    X, Y = np.meshgrid(x, y)
    
    # Function to check if a point is inside any circle
    def inside_circle(x, y, center_x, center_y, radius):
        return (x - center_x)**2 + (y - center_y)**2 <= radius**2

    # Create binary mask
    mask = np.logical_or(
        inside_circle(X, Y, circle1_x, circle1_y, circle1_radius),
        inside_circle(X, Y, circle2_x, circle2_y, circle2_radius)
    ).astype(int)


    # Create the tensor with two channels
    tensor = np.stack(mask, axis=0)

    return tensor

# @click.command()
# @click.option("--outdir", help="Where to save the output tensors", type=str, required=True)
# @click.option("--seed", help="Random seed", type=int, required=True)
# @click.option("--lres", "lres_path", help="Low-res network pickle path/URL", type=str, required=True)
# @click.option("--sres", "sres_path", help="Super-res network pickle path/URL", type=str)
# @click.option("--len", "seq_length", help="Video length in frames", type=int, default=301)
def generate_tensors_cond(
    outdir: str,
    seed: int,
    lres_path: str,
    seq_length: int,
    sres_path: str = None, # if path to super-res generator is not provided only low-res tensors are generated
    save_dict: bool = False,
    lres_cond: dict = None,
    sres_cond: dict = None,
    lres_G: torch.nn.Module = None,
    sres_G: torch.nn.Module = None, 
):
    """Generate tensors using pretrained model pickles.

        Example:

        gan_tensors = generate_tensors_cond(
                    outdir=./outdir_path,
                    seed=49,
                    lres_path=./lres/checkpoints/ckpt-00700000-train.pkl,  # path to  lres generator
                    sres_path=./sres/checkpoints/ckpt-00300000-train.pkl,  # path to  sres generator
                    seq_length=256, 
                    lres_cond=dict(cond_sp=lres_cond_sp, cond_num=lres_cond_num), # lres conditioning information
                    sres_cond=dict(cond_sp=sres_cond_sp, cond_num=sres_cond_num), # sres conditioning information
                )
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    lres_cond = {key: value.contiguous().to(device).to(torch.float32) for key, value in lres_cond.items()}
    sres_cond = sres_cond.to(device).to(torch.float32) if sres_cond is not None else None


    if lres_G is None:
        lres_G = utils.load_G(lres_path)
    if sres_G is None:
        sres_G = None if sres_path is None else utils.load_G(sres_path)

    print("Generating lr frames...")
    segment_length = 16
    lr_seq_length = ((seq_length + segment_length - 1) // segment_length) * segment_length
    lr_seq_length = lr_seq_length if sres_path is None else lr_seq_length + 2 * sres_G.temporal_context
    generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)


    start_time = time.perf_counter()
    lr_video = lres_G(1, lr_seq_length, cond=lres_cond, generator_emb=generator)
    elapsed = time.perf_counter() - start_time
    print(f"l_res inference took {elapsed:.5f} seconds")
    

    
    lr_video_tmp = lr_video.squeeze(0)

    if save_dict:
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    output_dict = {"lr_video": lr_video_tmp}

    if sres_path is not None:
       
        print("Generating hr frames...") 

        # Returns an iterator over segments, which enables efficiently handling long videos.
        if sres_cond is not None:

            
            if sres_cond.ndim == 3:
                # shape is (1, 256, 256) → add channel dim
                sres_cond = sres_cond.unsqueeze(0)

            start_time = time.perf_counter()
            segments = sres_G.sample_video_segments(lr_video,sres_cond, segment_length, generator_z=generator)
            elapsed = time.perf_counter() - start_time
            print(f"s_res inference took {elapsed:.5f} seconds")
        else:
            start_time = time.perf_counter()
            segments = sres_G.sample_video_segments(lr_video, segment_length, generator_z=generator)
            elapsed = time.perf_counter() - start_time
            print(f"s_res inference took {elapsed:.5f} seconds")
            
                       

    
        video = torch.cat(list(segments), dim=2)[:, :, :seq_length]

        lr_video = lr_video[:, :, sres_G.temporal_context : sres_G.temporal_context + seq_length]

        video = video.squeeze(0)
        lr_video = lr_video.squeeze(0)
        output_dict = {"lr_video": lr_video, "hr_video": video}
        



    if save_dict:
        # Move all tensors in output_dict to cpu before saving
        output_dict_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in output_dict.items()}
        output_path = pathlib.Path(outdir) / "output_dict.pt"
        torch.save(output_dict_cpu, output_path)
        print(f"Saved output_dict to {output_path}")

    return output_dict



