#!/usr/bin/python
# encoding: utf-8

import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .image_multi import load_data_detection
from .utils_multi import read_truths_args


class listDataset(Dataset):
    def __init__(
        self,
        input_file_list: Union[str, Path],
        label_file_list: Optional[Union[str, Path]] = None,
        shape=None,
        shuffle=True,
        transform=None,
        objclass=None,
        target_transform=None,
        train=False,
        seen=0,
        batch_size=64,
        num_workers=4,
        cell_size=32,
        bg_file_names=None,
        num_keypoints=9,
        max_num_gt=50,
        ssp=False,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bg_file_names = bg_file_names
        self.objclass = objclass
        self.cell_size = cell_size
        self.num_keypoints = num_keypoints
        # maximum number of ground-truth labels an image can have
        self.max_num_gt = max_num_gt
        self.ssp = ssp

        # TODO - should just have a kwarg that gives you the directory these paths are relative to
        rel_dir = Path(__file__).parent

        with open(input_file_list) as file:
            input_paths = [
                str((rel_dir / f).resolve()).strip() for f in file.readlines()
            ]

        if label_file_list:
            with open(label_file_list) as file:
                label_paths = [
                    str((rel_dir / f).resolve()).strip() for f in file.readlines()
                ]
        else:
            if train:
                # the actual labels are created in image_multi in a function that does augmentation,
                # so these "files" aren't used
                label_paths = [""] * len(input_paths)
            else:
                label_paths = [
                    input_path.replace("benchvise", self.objclass)
                    .replace("images", "labels_occlusion")
                    .replace("JPEGImages", "labels_occlusion")
                    .replace(".jpg", ".txt")
                    .replace(".png", ".txt")
                    for input_path in input_paths
                ]

        # [(input_path, label_path)]
        self.paths = list(zip(input_paths, label_paths))
        self.nSamples = len(self.paths)
        self.nbatches = self.nSamples // self.batch_size + 1

        if shuffle:
            random.shuffle(self.paths)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError(
                f"Index {index} is greater than the maximum index of {len(self) - 1}."
            )
        imgpath, label_path = self.paths[index]
        if self.ssp:
            img = Image.open(imgpath).convert("RGB")
            if self.shape:
                img = img.resize(self.shape)

            # +2 for ground-truth of width/height, +1 for class label
            num_labels = 2 * self.num_keypoints + 3
            label = torch.zeros(self.max_num_gt * num_labels)
            label_prefix = torch.from_numpy(np.load(label_path))
            label[: len(label_prefix)] = label_prefix

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                label = self.target_transform(label)

            self.seen = self.seen + self.num_workers
            return (img, label)

        if self.train and index % self.batch_size == 0:
            if self.seen < 20 * self.nbatches * self.batch_size:
                width = 13 * self.cell_size
                self.shape = (width, width)
            elif self.seen < 40 * self.nbatches * self.batch_size:
                width = (random.randint(0, 3) + 13) * self.cell_size
                self.shape = (width, width)
            elif self.seen < 60 * self.nbatches * self.batch_size:
                width = (random.randint(0, 5) + 12) * self.cell_size
                self.shape = (width, width)
            elif self.seen < 80 * self.nbatches * self.batch_size:
                width = (random.randint(0, 7) + 11) * self.cell_size
                self.shape = (width, width)
            else:
                width = (random.randint(0, 9) + 10) * self.cell_size
                self.shape = (width, width)

        if self.train:
            # Decide on how much data augmentation you are going to apply
            jitter = 0.1
            hue = 0.05
            saturation = 1.5
            exposure = 1.5

            # Get background image path
            random_bg_index = random.randint(0, len(self.bg_file_names) - 1)
            bgpath = self.bg_file_names[random_bg_index]

            img, label = load_data_detection(
                imgpath,
                self.shape,
                jitter,
                hue,
                saturation,
                exposure,
                bgpath,
                self.num_keypoints,
                self.max_num_gt,
            )
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert("RGB")
            if self.shape:
                img = img.resize(self.shape)

            # +2 for ground-truth of width/height, +1 for class label
            num_labels = 2 * self.num_keypoints + 3
            label = torch.zeros(self.max_num_gt * num_labels)
            if os.path.getsize(label_path):
                ow, oh = img.size
                tmp = torch.from_numpy(read_truths_args(label_path))
                tmp = tmp.view(-1)
                tsz = tmp.numel()
                if tsz > self.max_num_gt * num_labels:
                    label = tmp[0 : self.max_num_gt * num_labels]
                elif tsz > 0:
                    label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)
