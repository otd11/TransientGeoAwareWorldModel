
## Learning Transient Convective Heat Transfer with Geometry Aware World Models 

**Abstract.**  
Partial differential equation (PDE) simulations are fundamental to engineering and physics but are often computationally prohibitive for real-time applications. While generative AI offers a promising avenue for surrogate modeling, standard video generation architectures lack the specific control and data compatibility required for physical simulations. This paper introduces a geometry aware world model architecture, derived from a video generation architecture (LongVideoGAN), designed to learn transient physics. We introduce two key architecture elements: (1) a twofold conditioning mechanism incorporating global physical parameters and local geometric masks, and (2) an architectural adaptation to support arbitrary channel dimensions, moving beyond standard RGB constraints. We evaluate this approach on a 2D transient computational fluid dynamics (CFD) problem involving convective heat transfer from buoyancy-driven flow coupled to a heat flow in a solid structure. We demonstrate that the conditioned model successfully reproduces complex temporal dynamics and spatial correlations of the training data. Furthermore, we assess the model's generalization capabilities on unseen geometric configurations, highlighting both its potential for controlled simulation synthesis and current limitations in spatial precision for out-of-distribution samples.

**Code overview.**  
This repository provides the official PyTorch implementation of the method
described above. It extends LongVideoGAN with geometry aware conditioning and
supports dataset preparation, training, evaluation, and video generation for
transient convective heat transfer simulations.


## Requirements (same as **LongVideoGAN**)

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* * 1+ high-end NVIDIA GPU for synthesis. All experiments in the paper were conducted on A100 GPUs.
* CUDA toolkit 11.1 or later.
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your LongVideoGAN Python environment:
  - `conda env create -f environment.yml -n long-video-gan`
  - `conda activate long-video-gan`

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

See [StyleGAN3 troubleshooting](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md) for help on common installation and run-time problems.

## Generating videos (see also README of **LongVideoGAN**)

Pre-trained models are stored as `*.pkl` files. Make sure you have set up the required Python environment before generating videos.

You can use pre-trained networks in your own Python code. Please refer to the `generate_tensors_cond.py` file for a minimal example.

To run pretrained models requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle files directly load the `'G_ema'` network that is a moving average of the generator weights over several training steps. The network is a regular instance of `torch.nn.Module`, with all parameters and buffers placed on the CPU at import and gradient computation disabled by default.

Note that to generate very long videos without running out of memory, we recommend generating videos in chunks of shorter segments. In `generate.py`, we run the super-resolution network over in chunks by calling `sres_G.sample_video_segments(...)` rather than `sres_G(...)` directly. The low res network can similarly be called using `lres_G.sample_video_segments(...)` if you would like to generate longer videos and run into memory issues.

## Preparing datasets  (see also README of **LongVideoGAN**)

To create your own tensor dataset, see the `dataset_tools` directory. Make sure you have set up the required Python environment before creating the dataset. Then run the `dataset_tools/make_dataset_from_tensors` script. See the example below, and make sure to run the script separately for each shard/partition.

```.bash
python -m dataset_tools.make_dataset_from_tensors SOURCE_VIDEOS_DIR OUTPUT_DATASET_DIR \
    --height=256 --width=256 --partition=0 --num-partitions=10
```

Setting `--partition=0 --num-partitions=10` (default) in the above example will produce a single shard of the dataset as one ZIP archive containing roughly 1/10 of the videos. You must run the command separately for each partition from 0 to 9 to create all shards. For a small dataset you can set `--num-partitions=1` and for very large datasets you can set it to be larger than 10. Breaking the dataset into shards allows each shard to be created at the same time in parallel, or for creation of the shards to be distributed over different machines. See `dataset_tools/make_dataset_sbatch.sh` for an example of how to run dataset creation in parallel in a Slurm job.

## Training  (see also README of **LongVideoGAN**)

You can train new models using `train_lres.py` to train the low resolution network and `train_sres.py` to train the super-resolution network. We used 2 high-end NVIDIA GPUs for training. If that is not possible or you run out of memory, you can try increasing the number of gradient accumulation steps (`--grad-accum`), which will train more slowly but use less memory. You may also experiment with lowering the batch size (`--batch`), although this may worsen results. For multi-GPU training, we use [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

Distributed low-resolution training over 2 GPUs on an exemplary simulation dataset:
```.bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train_lres.py \
    --outdir=runs/lres --dataset=datasets/simulation --batch=8 --grad-accum=2 --gamma=1.0 
```

Distributed super-resolution training over 2 GPUs on an exemplary simulation dataset:
```.bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train_sres.py \
    --outdir=runs/sres --dataset=datasets/simulation --batch=8 --grad-accum=1 --gamma=1.0
```

Model checkpoints, random generated video samples, and metrics will be logged in a subdirectory of `--outdir` created for each run. Setting `--outdir` is optional and will default to the `runs/lres` and `runs/sres` directories shown above. We only support the most crucial arguments through the command line. Rather than rely on passing all arguments through the command line or reading a separate configuration file, other training settings can be modified directly in the train file, and the settings will be logged under the run subdirectory to `config.json` for each training run.

In `dataset.py`, conditioning information must be assembled into dictionaries (e.g., `lres_cond=dict(cond_sp=lres_cond_sp, cond_num=lres_cond_num)`) so that `train_lres.py` can correctly pass the conditioning signals to the GAN.

We use [W&B](https://wandb.ai/) for logging and recommend setting up an account to track experiments. If you prefer not to use W&B and have not already logged into a W&B account, select "Don't visualize my results" when prompted after launching a training run.


## License

This repository contains derivative work based on **LongVideoGAN**
Copyright (c) 2022, NVIDIA Corporation & affiliates.

The original NVIDIA Source Code License is included in `LICENSE.txt`
and applies to this repository.

## Citation

```
@inproceedings{doganay2026gawm,
    title={Learning Transient Convective Heat Transfer with Geometry Aware World Models},
    author={Doganay, Onur Tanil and Klawonn, Alexander and Eigel, Martin and Gottschalk, Hanno},
    year={2026},
    booktitle={Under review},
    }


```

## Acknowledgements
The authors thank Claudia Drygala, Francesca di Mare and Edmund Ross for interesting discussions. This work was funded by the German Research Council (DFG) through the center of excellence Math+ under the project PaA-5 "AI Based Simulation of Transient Physical Systems – From Benchmarks to Hybrid Solutions". Hanno Gottschalk also acknowledges financial support from the German research council through SPP2403 "Carnot Batteries" project GO 833/8-1 "Inverse aerodynamic design of turbo components for Carnot batteries by means of physics
informed networks enhanced by generative learning".

