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
from dataset import VideoDatasetTwoRes
from dnnlib import EasyDict
from metrics import metric_main
from model.video_gan_sres import SuperResVideoGAN
from torch_utils import distributed, misc, training_stats
from torch_utils.ops import bias_act, grid_sample_gradfix, upfirdn2d

# =====================================================================================================================


def train(
    *,
    run_dir: str,
    dataset_dir: str,
    seq_length: int,
    temporal_context: int,
    lr_height: int,
    lr_width: int,
    hr_height: int,
    hr_width: int,
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
    ada_interval: int,
    total_batch: int,
    metrics: list[str],
    metric_kwargs: EasyDict,
    loader_kwargs: EasyDict,
    gan_kwargs: EasyDict,
    channels: int,
    cond_sp_channels: int,
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

    with utils.context_timer0("Loading video datasets"):
        dataset = VideoDatasetTwoRes(
            dataset_dir,
            seq_length + 2 * temporal_context,
            lr_height,
            lr_width,
            hr_height,
            hr_width,
            x_flip=x_flip,
        )
        data_iter = utils.get_infinite_data_iter(
            dataset, batch_size=batch_per_gpu, seed=utils.random_seed(), **loader_kwargs
        )
        print_lr_video = next(data_iter)["lr_video"][:1].cuda()
        print_hr_cond_sp = next(data_iter)["hr_cond_sp"][:1].cuda()
        # print_lr_video = next(data_iter)["lr_video"][:1, :channels].cuda()
        # print_hr_cond_sp = next(data_iter)["hr_cond_sp"][:1, :channels].cuda()
        
        #debugging
        print(f"print_lr_video.shape {print_lr_video.shape}")
        print(f"print_hr_cond_sp.shape {print_hr_cond_sp.shape}")

    with utils.context_timer0("Saving real videos"):
        result_dataset = VideoDatasetTwoRes(
            dataset_dir,
            result_seq_length + 2 * temporal_context,
            lr_height,
            lr_width,
            hr_height,
            hr_width,
            x_flip=x_flip,
        )
        with torch.no_grad():
            generator = torch.Generator().manual_seed(seed_per_gpu)
            index = torch.randint(len(result_dataset) // world_size, (), generator=generator).item()
            index += rank * (len(result_dataset) // world_size)
            sample = result_dataset[index]
            result_lr_video = sample["lr_video"][None].cuda()
            result_hr_video = sample["hr_video"][None].cuda()
            
            # keep only the first N channels when preparing real videos
            # result_lr_video = sample["lr_video"][:channels][None].cuda()
            # result_hr_video = sample["hr_video"][:channels][None].cuda()
            
            # debugging
            print(f"result_lr_video.shape {result_lr_video.shape}")
            print(f"result_hr_video.shape {result_hr_video.shape}")
            
            
            
            # lr_cond_num = sample["lr_cond_num"][None].cuda() if "lr_cond_num" in sample else None
            # lr_cond_sp = sample["lr_cond_sp"][None].cuda() if "lr_cond_sp" in sample else None
            result_hr_cond_num = sample["hr_cond_num"][None].cuda() if "hr_cond_num" in sample else None
            result_hr_cond_sp = sample["hr_cond_sp"][None].cuda() if "hr_cond_sp" in sample else None
            hr_cond_dict = dict(cond_sp=result_hr_cond_sp, cond_num=result_hr_cond_num)

            lr_path = samples_dir.joinpath("real-lr.mp4") if rank == 0 else None
            hr_path = samples_dir.joinpath("real-hr.mp4") if rank == 0 else None
            # utils.write_video_grid(result_lr_video, lr_path, gather=True)
            # utils.write_video_grid(result_hr_video, hr_path, gather=True)
            # utils.write_video_for_each_channel(result_lr_video, lr_path, gather=True, channels=channels, onlyT=False) #, cond = dict(cond_sp=lr_cond_sp, cond_num=lr_cond_num))
            # utils.write_video_for_each_channel(result_hr_video, hr_path, gather=True, channels=channels, cond=hr_cond_dict, onlyT=False) #, cond = dict(cond_sp=hr_cond_sp, cond_num=hr_cond_num))


    with utils.context_timer0("Constructing GAN model"):
        
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

            video_gan = SuperResVideoGAN(
                    seq_length, temporal_context, lr_height, lr_width, hr_height, hr_width, **gan_kwargs
                )

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
            video_gan = SuperResVideoGAN(
                    seq_length, temporal_context, lr_height, lr_width, hr_height, hr_width, **gan_kwargs
                )
        
        


    # with utils.context_timer0("Constructing GAN model"):
    #     video_gan = SuperResVideoGAN(
    #         seq_length, temporal_context, lr_height, lr_width, hr_height, hr_width, **gan_kwargs
    #     )

    #     # Load checkpoint if resume is provided
    #     if resume is not None:
    #         utils.print0(f"Resuming from checkpoint: {resume}")
    #         with open(resume, "rb") as f:
    #             ckpt = pickle.load(f)
    #         # Handle both object and dict checkpoint formats
    #         if hasattr(ckpt, "G") and hasattr(ckpt, "D"):
    #             video_gan.G.load_state_dict(ckpt.G.state_dict())
    #             video_gan.D.load_state_dict(ckpt.D.state_dict())
    #             if hasattr(ckpt, "G_opt") and hasattr(ckpt, "D_opt"):
    #                 video_gan.G_opt.load_state_dict(ckpt.G_opt)
    #                 video_gan.D_opt.load_state_dict(ckpt.D_opt)
    #             if hasattr(ckpt, "G_ema"):
    #                 video_gan.G_ema.load_state_dict(ckpt.G_ema.state_dict())
    #         elif isinstance(ckpt, dict) and "G" in ckpt and "D" in ckpt:
    #             video_gan.G.load_state_dict(ckpt["G"].state_dict())
    #             video_gan.D.load_state_dict(ckpt["D"].state_dict())
    #             if "G_opt" in ckpt and "D_opt" in ckpt:
    #                 video_gan.G_opt.load_state_dict(ckpt["G_opt"])
    #                 video_gan.D_opt.load_state_dict(ckpt["D_opt"])
    #             if "G_ema" in ckpt:
    #                 video_gan.G_ema.load_state_dict(ckpt["G_ema"].state_dict())
    #         else:
    #             raise RuntimeError("Checkpoint does not contain G and D (and optionally G_opt, D_opt, G_ema).")




    stats_jsonl_fp = None
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time

    utils.print0(f"Training for steps {start_step:,} - {total_steps:,}\n")
    for step in range(start_step, total_steps + 1):
        onlyT=True
        
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
                    wandb.log(stats, step=step)

            if tick % ticks_per_G_ema_ckpt == 0:
                # Prints summaries of network architectures.
                if rank == 0:
                    with torch.inference_mode():
                        hr_video = misc.print_module_summary(video_gan.G, [print_lr_video, print_hr_cond_sp])
                        misc.print_module_summary(video_gan.D, [video_gan.crop_to_seq_length(print_lr_video), hr_video, print_hr_cond_sp])

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
                    generator = torch.Generator("cuda").manual_seed(seed_per_gpu)
                    segments = G_ema_ckpt.sample_video_segments(result_lr_video, result_hr_cond_sp, generator_z=generator)
                    path = samples_dir.joinpath(f"fake-{step:08d}-hr.mp4") if rank == 0 else None
                    # utils.write_video_grid(segments, path, gather=True)
                    # utils.write_video_for_each_channel(segments, path, gather=True, channels=channels, cond=hr_cond_dict, onlyT=onlyT) #, cond = dict(cond_sp=hr_cond_sp, cond_num=hr_cond_num))


                # Evaluates metrics.
                if len(metrics) > 0:
                    utils.print0(f"Evaluating metrics...")
                    wandb_results = dict()
                    # Default sequence length of 1 is overwritten by video metrics.
                    dataset_kwargs = dict(dataset_dir=dataset_dir, seq_length=1, height=hr_height, width=hr_width)
                    cond_dataset_kwargs = dict(dataset_dir=dataset_dir, seq_length=1, height=lr_height, width=lr_width)
                    for metric in metrics:
                        result_dict = metric_main.calc_metric(
                            metric=metric,
                            G=G_ema_ckpt,
                            dataset_kwargs=dataset_kwargs,
                            cond_dataset_kwargs=cond_dataset_kwargs,
                            **metric_kwargs,
                        )
                        if rank == 0:
                            json_line = json.dumps(dict(result_dict, step=step, G_ema_ckpt_path=str(G_ema_ckpt_path)))
                            print(json_line)
                            with open(Path(run_dir, f"metric-{metric}.jsonl"), "at") as fp:
                                fp.write(f"{json_line}\n")
                            for name, value in result_dict.results.items():
                                wandb_results[f"metric/{name}"] = value
                    if rank == 0:
                        wandb.log(wandb_results, step=step, commit=True)
                del G_ema_ckpt

            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

        if step == total_steps:
            utils.print0("Finished training!")
            break

        video_gan.update_lrates(step)

        # # Generator.
        # lr_video = next(data_iter)["lr_video"].cuda()
        # video_gan.update_G(lr_video, cond_sp)

        # # Discriminator.
        # sample = next(data_iter)
        # lr_video = sample["lr_video"].cuda()
        # hr_video = video_gan.crop_to_seq_length(sample["hr_video"]).cuda()
        # video_gan.update_D(lr_video, lr_video, hr_video, cond_sp)


        # added for super-resolution training
        sample = next(data_iter)
        lr_video = sample["lr_video"].cuda()
        hr_video = video_gan.crop_to_seq_length(sample["hr_video"]).cuda()
        hr_cond_sp = sample["hr_cond_sp"].cuda()
        

        
        # keep only the first N channels from the dataset (C dimension)
        # lr_video = sample["lr_video"][:, :channels].cuda()
        # hr_video = video_gan.crop_to_seq_length(sample["hr_video"][:, :channels]).cuda()
        # hr_cond_sp = sample["hr_cond_sp"].cuda()  # conditioning map can keep its full channel count
        
        # # debugging
        print(f"lr_video.shape {lr_video.shape}")
        print(f"hr_video.shape {hr_video.shape}")
        print(f"hr_cond_sp.shape {hr_cond_sp.shape}")
                


        video_gan.update_G(lr_video, hr_cond_sp)
        video_gan.update_D(lr_video, lr_video, hr_video, hr_cond_sp)


        # R1 regularization.
        if step % r1_interval == 0:
            sample = next(data_iter)
            lr_video = video_gan.crop_to_seq_length(sample["lr_video"]).cuda()
            hr_video = video_gan.crop_to_seq_length(sample["hr_video"]).cuda()
            hr_cond_sp = sample["hr_cond_sp"].cuda()
            video_gan.update_r1(lr_video, hr_video, hr_cond_sp, gain=r1_interval)
        
        # if step % r1_interval == 0:
        #     sample = next(data_iter)
        #     lr_video = video_gan.crop_to_seq_length(sample["lr_video"][:, :channels]).cuda()
        #     hr_video = video_gan.crop_to_seq_length(sample["hr_video"][:, :channels]).cuda()
        #     hr_cond_sp = sample["hr_cond_sp"].cuda()
        #     video_gan.update_r1(lr_video, hr_video, hr_cond_sp, gain=r1_interval)

        # Adaptive augmentation.
        if step % ada_interval == 0:
            video_gan.update_ada(gain=ada_interval)

        video_gan.update_G_ema(step)


# =====================================================================================================================


@click.command()
@click.option("--outdir", help="Where to make the output run directory", type=str, default="runs/sres")
@click.option("--dataset", "dataset_dir", help="Path to dataset directory", type=str, required=True)
@click.option("--batch", "total_batch", help="Total batch size across all GPUs and gradient accumulation steps", type=int, default=32)  # fmt: skip
@click.option("--grad-accum", help="Gradient accumulation steps", type=int, default=1)
@click.option("--gamma", "r1_gamma", help="R1 regularization gamma", type=float, default=1.0)
@click.option("--metric", "-m", "metrics", help="Metrics to compute", default=[], type=str, multiple=True)
@click.option("--channels", help="Amount of channels", type=int, default=3)
@click.option("--resume", help="Path to training checkpoint (.pkl) to resume from", type=str, default=None)
@click.option("--start-step", help="Step to start training from", type=int, default=0)
@click.option("--cond-sp-dim", help="Number of channels in the conditioning spatial feature map", type=int, default=1)
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
    cond_sp_dim = 1,
):
    """Train a super-resolution LongVideoGAN network.
    Example:

    \b
    # Distributed super-resolution training over 8 GPUs on horseback riding dataset.
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 train_sres.py \\
        --outdir=runs/sres --dataset=datasets/horseback --batch=32 --grad-accum=1 --gamma=1.0 --metric=fvd2048_16f 
    """
    c = EasyDict(
        run_dir=None,
        dataset_dir=dataset_dir,
        seq_length=4,
        temporal_context=4,
        lr_height=64,
        lr_width=64,
        hr_height=256,
        hr_width=256,
        x_flip=False,
        seed=None,
        benchmark=False,
        allow_fp16_reduce=False,
        allow_tf32=True,
        start_step=start_step,
        # total_steps=100, #  for testing
        # steps_per_tick=1,
        # ticks_per_G_ema_ckpt=1,
        # ticks_per_train_ckpt=1,
        total_steps=300000, #275000,
        steps_per_tick=500,
        ticks_per_G_ema_ckpt=10,
        ticks_per_train_ckpt=100,
        result_seq_length=256,
        r1_interval=16,
        ada_interval=4,
        total_batch=total_batch,
        metrics=metrics,
        channels=channels,
        cond_sp_channels=cond_sp_dim,  # Number of channels in the conditioning spatial feature map
        resume=resume,

        # run_dir=None,
        # dataset_dir=dataset_dir,
        # seq_length=4,
        # temporal_context=4,
        # lr_height=36,
        # lr_width=64,
        # hr_height=144,
        # hr_width=256,
        # x_flip=True,
        # seed=None,
        # benchmark=False,
        # allow_fp16_reduce=False,
        # allow_tf32=False,
        # start_step=0,
        # total_steps=275000,
        # steps_per_tick=500,
        # ticks_per_G_ema_ckpt=10,
        # ticks_per_train_ckpt=100,
        # result_seq_length=256,
        # r1_interval=16,
        # ada_interval=4,
        # total_batch=total_batch,
        # metrics=metrics,
    )

    c.loader_kwargs = EasyDict(
        num_workers=1,
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
        D_lrate=0.003,
        D_beta2=0.99,
        lr_cond_prob=0.1,
        r1_gamma=r1_gamma,
        in_augment_p=0, # since we are dealing with physics disable augmentation 
        in_augment_strength=0, # since we are dealing with physics disable augmentation 
        augment_real_sign_target = None, # Has default value in SuperResVideoGAN. To use augmentation comment out this line.
        # in_augment_p=0.5, 
        # in_augment_strength=8,
        G_grad_accum=grad_accum,
        D_grad_accum=grad_accum,
        channels=channels, #
    )

    if c.r1_interval > 0:
        mb_ratio = c.r1_interval / (c.r1_interval + 1)
        c.gan_kwargs.D_lrate *= mb_ratio
        c.gan_kwargs.D_beta2 **= mb_ratio

    c.gan_kwargs.G_kwargs = EasyDict(
        class_name="model.generator_sres.VideoGenerator",
        num_fp16_res=4,
        fourfeats=False,
        channels=channels, #
        cond_sp_channels=c.cond_sp_channels,  # Number of channels in the conditioning spatial feature map
    )
    c.gan_kwargs.D_kwargs = EasyDict(
        class_name="model.discriminator_sres.VideoDiscriminator",
        num_fp16_res=4,
        cond_sp_channels=c.cond_sp_channels,  # Number of channels in the conditioning spatial feature map
    )

    c.gan_kwargs.augment_kwargs = EasyDict(
        xflip=1,
        rotate90=1,
        xint=1,
        scale=1,
        rotate=1,
        aniso=1,
        xfrac=1,
        brightness=1,
        contrast=1,
        lumaflip=1,
        hue=1,
        saturation=1,
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
            wandb.init(
                dir=c.run_dir,
                name=Path(c.run_dir).name,
                project="long-video-gan-sres",
                config=c,
                settings=wandb.Settings(start_method="spawn"),
            )

        train(**c)


# =====================================================================================================================


if __name__ == "__main__":
    main()
