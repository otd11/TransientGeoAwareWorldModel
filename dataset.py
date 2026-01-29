# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional
from zipfile import ZipFile

import einops
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# =====================================================================================================================


@dataclass
class VideoDataset(Dataset):
    dataset_dir: str
    seq_length: int
    height: int
    width: int
    min_spacing: int = 1
    max_spacing: int = 1
    min_video_length: Optional[int] = None
    x_flip: bool = False

    def __post_init__(self):
        assert self.seq_length >= 1

        self.dataset_path = Path(self.dataset_dir).joinpath(f"{self.height:04d}x{self.width:04d}")
        assert self.dataset_path.is_dir(), self.dataset_path

        self.frame_paths = {}
        for partition in self.dataset_path.glob("*.zip"):
            with ZipFile(partition) as zf:
                with zf.open("frame_paths.json", "r") as fp:
                    self.frame_paths[partition.stem] = json.load(fp)

        self.min_video_length = max(self.min_video_length or 1, (self.seq_length - 1) * self.min_spacing + 1)

        self.video_paths = [
            (partition_name, clip_path, frame_names)
            for partition_name, partition_frame_paths in sorted(self.frame_paths.items())
            for clip_path, frame_names in sorted(partition_frame_paths.items())
            if len(frame_names) >= self.min_video_length
        ]

        self._zipfiles = {}

    def sample_frame_names(self, frame_names: list[str]) -> tuple[list[str], int]:
        max_spacing = (
            1 if self.seq_length == 1 else min(self.max_spacing, (len(frame_names) - 1) // (self.seq_length - 1))
        )
        spacing = torch.randint(self.min_spacing, max_spacing + 1, size=()).item()

        frame_span = (self.seq_length - 1) * spacing + 1
        max_start_index = len(frame_names) - frame_span
        start_index = torch.randint(max_start_index + 1, size=()).item()

        frame_names = frame_names[start_index : start_index + frame_span : spacing]
        return frame_names, spacing


    def read_frame(self, partition_name: str, frame_path: str) -> torch.Tensor:
        if partition_name not in self._zipfiles:
            partition_path = self.dataset_path.joinpath(f"{partition_name}.zip")
            self._zipfiles[partition_name] = ZipFile(partition_path)

        with self._zipfiles[partition_name].open(frame_path, "r") as fp:
            frame_data = torch.load(fp)  # Load tensor "frame" or dictionary with "frame" and optionally "cond"


            if isinstance(frame_data, torch.Tensor):
                frame = frame_data
            else:
                frame = frame_data["frame"]           
                frame = frame.to(torch.float32)  

                cond_sp = frame_data["cond_sp"]
                cond_sp = cond_sp.to(torch.float32) 

                cond_num = frame_data["cond_num"]
                cond_num = cond_num.to(torch.float32)
                
                # Ensure cond_sp has shape (c, h, w) by adding a channel dimension if necessary
                if cond_sp.dim() == 2:  # Check if cond_sp has shape (h, w)
                    cond_sp = cond_sp.unsqueeze(0)  # Add a channel dimension to make it (1, h, w)
                

           




        ###########################################################################
        ## Normalization of the dataset
        

        # rescale [T_min, T_max] to [0, 1] via  (T - 299)/15
        # rescale [u_min, u_max] to [0, 1] via (u + 0.2)/0.4
        # rescale [v_min, v_max] to [0, 1] via (v + 0.16)/0.4
        # rescale [p_min, p_max] to [0, 1] via (p + 6.3)/13
        

        # Normalize each channel        
        frame[0] = (frame[0] - 299)/ 15  
        frame[1] = (frame[1] + 0.2) / 0.4 
        frame[2] = (frame[2] + 0.16) / 0.4
        frame[3] = (frame[3] + 6.3) / 13.0

        # Normalize T_amb in cond_num
        cond_num[0] = (cond_num[0] - 299) / 15 # normalize T_amb

        if cond_num.shape[0] > 1:
            cond_num[1] = cond_num[1] / 100 # normalize Q_a, Q \in [0, 100] 
            cond_num[2] = cond_num[2] / 100 # normalize Q_b, Q \in [0, 100] 
        

        return dict(frame=frame, cond_sp=cond_sp, cond_num=cond_num)

    def __getitem__(self, index: int) -> dict[str, Any]:
        partition_name, clip_path, frame_names = self.video_paths[index]
        frame_names, spacing = self.sample_frame_names(frame_names)
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in frame_names]
        
        frames = []
        conds_sp = []
        cond_sp = []     
        conds_num = []
        cond_num = [] 

        for frame_path in frame_paths:
            frame_data = self.read_frame(partition_name, frame_path)  # Returns a dictionary with "frame" and optionally "cond"
            

            
            frames.append(frame_data["frame"])         
            conds_sp.append(frame_data["cond_sp"]) 
            conds_num.append(frame_data["cond_num"]) 
        
        video = torch.stack(frames, dim=1)  # Stack along the temporal dimension (T)

       
        
        cond_sp = conds_sp[0] # take cond_sp from the first frame
        cond_num = conds_num[0]

        
        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))
            cond_sp = cond_sp.flip(dims=(-1,))

        cond_dict = dict(cond_sp=cond_sp, cond_num=cond_num)
        return_dict = dict(video=video, cond=cond_dict, spacing=spacing)
    
        return return_dict

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getstate__(self):
        return dict(self.__dict__, _zipfiles={})


# =====================================================================================================================


@dataclass
class VideoDatasetTwoRes(Dataset):
    dataset_dir: str
    seq_length: int
    lr_height: int
    lr_width: int
    hr_height: int
    hr_width: int
    min_spacing: int = 1
    max_spacing: int = 1
    min_video_length: Optional[int] = None
    x_flip: bool = False

    def __post_init__(self):
        self.lr_dataset = VideoDataset(
            self.dataset_dir,
            self.seq_length,
            self.lr_height,
            self.lr_width,
            self.min_spacing,
            self.max_spacing,
            self.min_video_length,
            x_flip=self.x_flip,
        )
        self.hr_dataset = VideoDataset(
            self.dataset_dir,
            self.seq_length,
            self.hr_height,
            self.hr_width,
            self.min_spacing,
            self.max_spacing,
            self.min_video_length,
            x_flip=self.x_flip,
        )
        assert self.lr_dataset.video_paths == self.hr_dataset.video_paths

    def __getitem__(self, index: int) -> dict[str, Any]:
        partition_name, clip_path, frame_names = self.lr_dataset.video_paths[index]
        frame_names, spacing = self.lr_dataset.sample_frame_names(frame_names)
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in frame_names]


        lr_frames = [self.lr_dataset.read_frame(partition_name, frame_path)["frame"] for frame_path in frame_paths]  
        hr_frames = [self.hr_dataset.read_frame(partition_name, frame_path)["frame"] for frame_path in frame_paths]
        hr_cond_num = self.hr_dataset.read_frame(partition_name, frame_paths[0])["cond_num"]
        hr_cond_sp = self.hr_dataset.read_frame(partition_name, frame_paths[0])["cond_sp"]
        lr_video = torch.stack(lr_frames, dim=1)
        hr_video = torch.stack(hr_frames, dim=1)

        if self.x_flip and torch.rand(()).item() < 0.5:
            lr_video = lr_video.flip(dims=(-1,))
            hr_video = hr_video.flip(dims=(-1,))

        return dict(lr_video=lr_video, hr_video=hr_video, hr_cond_sp=hr_cond_sp, hr_cond_num=hr_cond_num, spacing=spacing)

    def __len__(self) -> int:
        return len(self.lr_dataset)


# =====================================================================================================================


@dataclass
class VideoDatasetPerImage(Dataset):
    dataset_dir: str
    height: int
    width: int
    seq_length: int = 1
    x_flip: bool = False

    def __post_init__(self):
        self.dataset = VideoDataset(self.dataset_dir, seq_length=1, height=self.height, width=self.width)

        self.video_paths = []
        for partition_name, partition_frame_paths in sorted(self.dataset.frame_paths.items()):
            for clip_path, frame_names in sorted(partition_frame_paths.items()):
                num_samples_from_source = len(frame_names) - self.seq_length + 1
                for start_index in range(0, num_samples_from_source):
                    sample_frame_names = frame_names[start_index : start_index + self.seq_length]
                    self.video_paths.append((partition_name, clip_path, sample_frame_names, num_samples_from_source))

    def __getitem__(self, index: int) -> dict[str, Any]:
        partition_name, clip_path, sample_frame_names, num_samples_from_source = self.video_paths[index]
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in sample_frame_names]
        frames = [self.dataset.read_frame(partition_name, frame_path) for frame_path in frame_paths]
        video = torch.stack(frames, dim=1)

        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))

        return dict(video=video, num_samples_from_source=num_samples_from_source)

    def __len__(self) -> int:
        return len(self.video_paths)


# =====================================================================================================================
