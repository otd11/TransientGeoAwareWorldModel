# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import click
import psutil
import torch
import torch.distributed as dist
import wandb

import dnnlib
import utils
from dataset import VideoDataset
from dnnlib import EasyDict
from metrics import metric_main
from model.video_gan_lres import LowResVideoGAN
from torch_utils import distributed, misc, training_stats
from torch_utils.ops import bias_act, grid_sample_gradfix, upfirdn2d

# =====================================================================================================================


def train(
    *,
    run_dir: str,
    dataset_dir: str,
    seq_length: int,
    height: int,
    width: int,
    x_flip: bool,
    seed: int,
    benchmark: bool,
    allow_fp16_reduce: bool,
    allow_tf32: bool,
    start_step: int,
    total_steps: int,
    steps_per_tick: int,
    ticks_per_G_ema_ckpt: int,
    ticks_per_train_ckpt: int,
    result_seq_length: int,
    r1_interval: int,
    total_batch: int,
    metrics: list[str],
    metric_kwargs: EasyDict,
    loader_kwargs: EasyDict,
    gan_kwargs: EasyDict,
    channels: int,
    cond_num_dim: int,
    cond_sp_dim: int,
    resume: str = None
):
    start_time = time.time()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()

    assert total_batch % world_size == 0, "Total batch size must be divisible by world size"
    assert ticks_per_train_ckpt % ticks_per_G_ema_ckpt == 0, "Invalid train checkpoint interval"
    batch_per_gpu = total_batch // world_size

    seed_per_gpu = rank + seed * world_size
    torch.manual_seed(seed_per_gpu)

    torch.backends.cudnn.benchmark = benchmark
    major, minor = torch.__version__.split(".")[:2]
    if int(major) > 1 or int(minor) > 10:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_fp16_reduce
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    # Initializes custom ops.
    grid_sample_gradfix.enabled = True
    bias_act._init()
    upfirdn2d._init()
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        ckpt_dir = Path(run_dir, "checkpoints")
        samples_dir = Path(run_dir, "samples")
        ckpt_dir.mkdir()
        samples_dir.mkdir()

    with utils.context_timer0("Loading video dataset"):
        result_dataset = VideoDataset(dataset_dir, result_seq_length, height, width, x_flip=x_flip)
        dataset = VideoDataset(dataset_dir, seq_length, height, width, x_flip=x_flip)
        data_iter = utils.get_infinite_data_iter(
            dataset, batch_size=batch_per_gpu, seed=utils.random_seed(), **loader_kwargs
        )






    with utils.context_timer0("Saving real videos"):
        generator = torch.Generator().manual_seed(seed_per_gpu)
        index = torch.randint(len(result_dataset) // world_size, (), generator=generator).item()
        index += rank * (len(result_dataset) // world_size)

        sample = result_dataset[index]



        video = sample["video"][None].cuda()
        cond_sp = sample["cond"]["cond_sp"][None].cuda()
        cond_num = sample["cond"]["cond_num"][None].cuda()
        cond = dict(cond_sp=cond_sp, cond_num=cond_num)


        path = samples_dir.joinpath("real-long.mp4") if rank == 0 else None


        generator = torch.Generator().manual_seed(seed_per_gpu)
        index = torch.randint(len(dataset) // world_size, (), generator=generator).item()
        index += rank * (len(dataset) // world_size)


        sample = dataset[index]

        video = sample["video"][None].cuda()
        cond_sp = sample["cond"]["cond_sp"][None].cuda()
        cond_num = sample["cond"]["cond_num"][None].cuda()
        cond = dict(cond_sp=cond_sp, cond_num=cond_num)

        path = samples_dir.joinpath("real-train.mp4") if rank == 0 else None



    with utils.context_timer0("Constructing low res GAN model"):
        # Load checkpoint if resume is provided
        if resume:
            ckpt_path = Path(resume)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {resume}")
            utils.print0(f"Resuming training from checkpoint: {resume}")

            # 1) load train and ema ckpt
            with open(ckpt_path, "rb") as fp:
                train_ckpt = pickle.load(fp)
            start_step = train_ckpt["step"]

            ema_path = ckpt_path.with_name(ckpt_path.name.replace("-train.pkl", "-G-ema.pkl"))
            with open(ema_path, "rb") as fp:
                G_ema_ckpt = pickle.load(fp)

            video_gan = LowResVideoGAN(seq_length, height, width, **gan_kwargs)

            # 2) overwrite its submodules wholesale
            # video_gan.G     = train_ckpt["G"]()       # instantiates and loads the saved G
            # video_gan.D     = train_ckpt["D"]()       # same for D
            # video_gan.G_ema = G_ema_ckpt()             # after loading G_ema_ckpt above

            video_gan.G.load_state_dict(train_ckpt["G"].state_dict())
            video_gan.D.load_state_dict(train_ckpt["D"].state_dict())
            video_gan.G_ema.load_state_dict(G_ema_ckpt.state_dict())


            # 3) rebuild optimizers around these exact parameters
            # video_gan.G_opt = OptimClass(video_gan.G.parameters(), **opt_kwargs)
            # video_gan.D_opt = OptimClass(video_gan.D.parameters(), **opt_kwargs)

            video_gan.G_opt.load_state_dict(train_ckpt["G_opt"])
            video_gan.D_opt.load_state_dict(train_ckpt["D_opt"])

            utils.print0(f"Resumed training from step {start_step}")
        else:
            video_gan = LowResVideoGAN(seq_length, height, width, **gan_kwargs)



    stats_jsonl_fp = None
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time

    utils.print0(f"Training for steps {start_step:,} - {total_steps:,}\n")
    for step in range(start_step, total_steps + 1):
        onlyT=True
        # print("train: step: ", step)

        # Extract "video" and "cond" from the next sample. This was originally done later in the loop.
        sample = next(data_iter)
        video = sample["video"].cuda()
        video = video[:, :channels]
          
        cond_sp = sample["cond"]["cond_sp"].cuda()
        cond_num = sample["cond"]["cond_num"].cuda()
        cond = dict(cond_sp=cond_sp, cond_num=cond_num)
        
        

        if step % steps_per_tick == 0:
            tick = step // steps_per_tick
            tick_end_time = time.time()

            # Accumulates training stats and prints status.
            if step > start_step:
                total_sec = tick_end_time - start_time
                sec_per_step = (tick_end_time - tick_start_time) / steps_per_tick

                cpu_mem_gb = psutil.Process(os.getpid()).memory_info().rss / 2**30
                peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / 2**30
                peak_gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / 2**30
                torch.cuda.reset_peak_memory_stats()

                status = (
                    f"step {training_stats.report0('progress/step', step):<8d} "
                    f"tick {training_stats.report0('progress/tick', tick):<5d} "
                    f"time {dnnlib.util.format_time(training_stats.report0('timing/total_sec', total_sec)):<12s} "
                    f"sec/step {training_stats.report0('timing/sec_per_step', sec_per_step):<7.2f} "
                    f"maintenance {training_stats.report0('timing/maintenance_sec', maintenance_time):<6.1f} "
                    f"cpumem {training_stats.report0('resources/cpu_mem_gb', cpu_mem_gb):<6.2f} "
                    f"gpumem {training_stats.report0('resources/peak_gpu_mem_gb', peak_gpu_mem_gb):<6.2f} "
                    f"reserved {training_stats.report0('resources/peak_gpu_mem_reserved_gb', peak_gpu_mem_reserved_gb):<6.2f} "
                )
                utils.print0(status)

                training_stats.default_collector.update()
                if rank == 0:
                    stats = training_stats.default_collector.as_dict()
                    if stats_jsonl_fp is None:
                        stats_jsonl_fp = open(Path(run_dir, "stats.jsonl"), "at")
                    stats_jsonl_fp.write(f"{json.dumps(dict(stats, timestamp=time.time()))}\n")
                    stats_jsonl_fp.flush()
                    stats = {name: value.mean for name, value in stats.items()}
                    # wandb.log(stats, step=step)

            if tick % ticks_per_G_ema_ckpt == 0:
                # Prints summaries of network architectures.
                if rank == 0:
                    with torch.inference_mode():
                        
                        cond_0 = dict(cond_sp=cond["cond_sp"][0], cond_num=cond["cond_num"][0])                                
                                  

                        video_tmp = misc.print_module_summary(video_gan.G, [1, seq_length], cond=cond_0) 
                        misc.print_module_summary(video_gan.D, [video_tmp], cond=cond_0) 
                        

                # Saves checkpoint.
                G_ema_ckpt, train_ckpt = video_gan.ckpt()
                train_ckpt["step"] = step  # Save the current step
                if rank == 0:
                    G_ema_ckpt_path = ckpt_dir.joinpath(f"ckpt-{step:08d}-G-ema.pkl")
                    with open(G_ema_ckpt_path, "wb") as fp:
                        pickle.dump(G_ema_ckpt, fp)

                    if tick % ticks_per_train_ckpt == 0:
                        onlyT=False # when saving a checkpoint plot all fields
                        
                        train_ckpt_path = ckpt_dir.joinpath(f"ckpt-{step:08d}-train.pkl")
                        with open(train_ckpt_path, "wb") as fp:
                            pickle.dump(train_ckpt, fp)
                        del train_ckpt

                # Saves generated video samples.
                with torch.no_grad():

                    cond_0 = dict(cond_sp=cond_sp[0].unsqueeze(0), cond_num=cond_num[0].unsqueeze(0))
                    

                    generator = torch.Generator("cuda").manual_seed(seed_per_gpu)
                    segments = G_ema_ckpt.sample_video_segments(1, result_seq_length, cond=cond_0, generator_emb=generator)
                    path = samples_dir.joinpath(f"fake-{step:08d}.mp4") if rank == 0 else None                                      
                    utils.write_video_for_each_channel(segments, path, gather=True, channels=channels, cond=cond, onlyT=onlyT)  


                # Evaluates metrics.
                if len(metrics) > 0:
                    utils.print0(f"Evaluating metrics...")
                    wandb_results = dict()
                    # Default sequence length of 1 is overwritten by video metrics.
                    dataset_kwargs = dict(dataset_dir=dataset_dir, seq_length=1, height=height, width=width)
                    metric_kwargs["replace_cache"] = metric_kwargs.get("replace_cache", False) and step == 0
                    for metric in metrics:
                        result_dict = metric_main.calc_metric(
                            metric=metric, G=G_ema_ckpt, dataset_kwargs=dataset_kwargs, **metric_kwargs
                        )
                        if rank == 0:
                            json_line = json.dumps(dict(result_dict, step=step, G_ema_ckpt_path=str(G_ema_ckpt_path)))
                            print(json_line)
                            with open(Path(run_dir, f"metric-{metric}.jsonl"), "at") as fp:
                                fp.write(f"{json_line}\n")
                            for name, value in result_dict.results.items():
                                wandb_results[f"metric/{name}"] = value
                    # if rank == 0:
                        # wandb.log(wandb_results, step=step, commit=True)
                del G_ema_ckpt

            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

        if step == total_steps:
            utils.print0("Finished training!")
            break

        video_gan.update_lrates(step)

        # Generator.
        video_gan.update_G(batch_per_gpu, cond)

        # Discriminator.
        video_gan.update_D(video, cond)

        # R1 regularization.
        if step % r1_interval == 0:
            sample = next(data_iter)
            video = sample["video"].cuda()
            cond_sp = sample["cond"]["cond_sp"].cuda()
            cond_num = sample["cond"]["cond_num"].cuda()
            cond = dict(cond_sp=cond_sp, cond_num=cond_num)
            video = video[:, :channels]

            video_gan.update_r1(video, cond=cond, gain=r1_interval)

        video_gan.update_G_ema(step)


# =====================================================================================================================


@click.command()
@click.option("--outdir", help="Where to make the output run directory", type=str, default="runs/lres")
@click.option("--dataset", "dataset_dir", help="Path to dataset directory", type=str, required=True)
@click.option("--batch", "total_batch", help="Total batch size across all GPUs and gradient accumulation steps", type=int, default=64)  # fmt: skip
@click.option("--grad-accum", help="Gradient accumulation steps", type=int, default=2)
@click.option("--gamma", "r1_gamma", help="R1 regularization gamma", type=float, default=1.0)
@click.option("--metric", "-m", "metrics", help="Metrics to compute", default=[], type=str, multiple=True)
@click.option("--channels", help="Amount of channels", type=int, default=3)
@click.option("--resume", help="Path to training checkpoint (.pkl) to resume from", type=str, default=None)
@click.option("--start-step", help="Step to start training from", type=int, default=0)
# @click.option("--cond-channel", help="Dimension of conditioning vector", type=int, default=0)
def main(
    outdir: str,
    dataset_dir: str,
    total_batch: int,
    grad_accum: int,
    r1_gamma: float,
    metrics: list[str],
    channels: int,
    resume: str = None,
    start_step: int = 0,
):
    """Train a low-resolution LongVideoGAN network.
    Example:

    \b
    # Distributed low-resolution training over 8 GPUs on horseback riding dataset.
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 train_lres.py \\
        --outdir=runs/lres --dataset=datasets/horseback --batch=64 --grad-accum=2 --gamma=1.0 --metric=fvd2048_128f
    """
    c = EasyDict(
        run_dir=None,
        dataset_dir=dataset_dir,
        seq_length=128,
        height=64,
        width=64,
        x_flip=False,
        seed=None,
        benchmark=False,
        allow_fp16_reduce=False,
        allow_tf32=True,
        start_step=start_step,
        total_steps=1000000,
        steps_per_tick=500,
        ticks_per_G_ema_ckpt=10,
        ticks_per_train_ckpt=100,
        result_seq_length=256,
        r1_interval=16,
        total_batch=total_batch,
        metrics=metrics,
        channels=channels,  # new option 
        resume=resume,    
        cond_num_dim=3,    
        cond_sp_dim=1,
    )

    c.loader_kwargs = EasyDict(
        num_workers=1, #2,
        prefetch_factor=2,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    c.metric_kwargs = EasyDict(
        normalize_weighting=True,
        single_sample_per_video=False,
        replace_cache=False,
        verbose=False,
    )

    c.gan_kwargs = EasyDict(
        D_lrate=0.002,
        D_beta2=0.99,
        r1_gamma=r1_gamma,
        G_random_temp_translate=True,
        temp_scale_augment=1.0,
        G_grad_accum=grad_accum,
        D_grad_accum=grad_accum,
        channels=channels, # amount of channels
       diffaug_policy = "" #"color,translation,cutout"    # disable augmentation
    )

    if c.r1_interval > 0:
        mb_ratio = c.r1_interval / (c.r1_interval + 1)
        c.gan_kwargs.D_lrate *= mb_ratio
        c.gan_kwargs.D_beta2 **= mb_ratio
        

    


    c.gan_kwargs.G_kwargs = EasyDict(
        class_name="model.generator_lres.VideoGenerator",
        num_fp16_layers=0,
        temporal_padding=8,
        temporal_emb_dim=1024,
        channels=channels, # amount of channels
        cond_num_dim=c.cond_num_dim,
        cond_sp_dim=c.cond_sp_dim,
        embedding_kwargs= {
            # "channels": 1024,
            "min_sampling_rate": 5.0, #250.0,     # Kaiser window ranges # start at 5 to obtain symmetric filter and avoid cutoff at 2.0 since 5/2>2.0
            "max_sampling_rate": 800.0, #10000.0,   # Kaiser window ranges # video have 720 frames in total
            # "blur_widths": 128,
            # "cutoff": 2.0,
            # "width": 12.0,
            # "sampling_rate_base": 2.0,
            # "normalize_per_filter": 1.0,
        }
    )
    c.gan_kwargs.D_kwargs = EasyDict(
        class_name="model.discriminator_lres.VideoDiscriminator",
        num_fp16_res=0,
        channels=channels, # amount of channels
        cond_num_dim=c.cond_num_dim,
        cond_sp_dim=c.cond_sp_dim,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        distributed.init(temp_dir)

        if distributed.get_rank() == 0:
            desc = f"{Path(c.dataset_dir).name}-{total_batch}batch-{grad_accum}accum-{r1_gamma}gamma"
            c.run_dir = utils.get_next_run_dir(outdir, desc=desc)
            Path(c.run_dir).mkdir(parents=True, exist_ok=True)

        # Sets random seed.
        if c.seed is None:
            c.seed = utils.random_seed()
        utils.print0(f"Random seed: {c.seed}")

        if distributed.get_rank() == 0:
            # Logs config to file.
            with open(Path(c.run_dir, "config.json"), "w") as fp:
                json.dump(c, fp, indent=2)

            # Initializes W&B.
            # wandb.init(
            #     dir=c.run_dir,
            #     name=Path(c.run_dir).name,
            #     project="long-video-gan-lres",
            #     config=c,
            #     settings=wandb.Settings(start_method="spawn"),
            # )

        train(**c)


# =====================================================================================================================


if __name__ == "__main__":
    main()
