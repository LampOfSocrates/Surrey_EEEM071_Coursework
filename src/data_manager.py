# Copyright (c) EEEM071, University of Surrey

import math
import random
from collections import defaultdict
from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset
from .datasets import init_imgreid_dataset
from .samplers import build_train_sampler
from .transforms import build_transforms


class BaseDataManager:
    def __init__(
        self,
        use_gpu,
        source_names,
        target_names,
        root="datasets",
        height=128,
        width=256,
        train_batch_size=32,
        test_batch_size=100,
        workers=4,
        train_sampler="",
        random_erase=False,  # use random erasing for data augmentation
        color_jitter=False,  # randomly change the brightness, contrast and saturation
        color_aug=False,  # randomly alter the intensities of RGB channels
        num_instances=4,  # number of instances per identity (for RandomIdentitySampler)
        data_fraction=1.0,  # fraction of dataset to use; 1.0 = full dataset
        **kwargs,
    ):
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.random_erase = random_erase
        self.color_jitter = color_jitter
        self.color_aug = color_aug
        self.num_instances = num_instances
        self.data_fraction = max(0.0, min(1.0, data_fraction))

        transform_train, transform_test = build_transforms(
            self.height,
            self.width,
            random_erase=self.random_erase,
            color_jitter=self.color_jitter,
            color_aug=self.color_aug,
        )
        self.transform_train = transform_train
        self.transform_test = transform_test

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return (
            self.testdataset_dict[name]["query"],
            self.testdataset_dict[name]["gallery"],
        )


class ImageDataManager(BaseDataManager):
    """
    Vehicle-ReID data manager
    """

    def __init__(self, use_gpu, source_names, target_names, **kwargs):
        super().__init__(use_gpu, source_names, target_names, **kwargs)

        print("=> Initializing TRAIN (source) datasets")
        train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_imgreid_dataset(root=self.root, name=name)

            for img_path, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                train.append((img_path, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        # --- data_fraction: subsample training set while keeping RandomIdentitySampler valid
        if self.data_fraction < 1.0:
            pid_to_imgs = defaultdict(list)
            for item in train:
                pid_to_imgs[item[1]].append(item)
            train = []
            for pid, imgs in pid_to_imgs.items():
                keep = max(self.num_instances, math.ceil(len(imgs) * self.data_fraction))
                train.extend(random.sample(imgs, min(keep, len(imgs))))
            print(f"  [data_fraction={self.data_fraction}] Train samples after subsample: {len(train)}")

        self.train_sampler = build_train_sampler(
            train,
            self.train_sampler,
            train_batch_size=self.train_batch_size,
            num_instances=self.num_instances,
        )
        self.trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train),
            sampler=self.train_sampler,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=self.use_gpu,
            drop_last=True,
        )

        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {
            name: {"query": None, "gallery": None} for name in target_names
        }
        self.testdataset_dict = {
            name: {"query": None, "gallery": None} for name in target_names
        }

        for name in self.target_names:
            dataset = init_imgreid_dataset(root=self.root, name=name)

            query   = list(dataset.query)
            gallery = list(dataset.gallery)

            # --- data_fraction: subsample query/gallery for faster evaluation
            if self.data_fraction < 1.0:
                nq = max(1, math.ceil(len(query) * self.data_fraction))
                ng = max(1, math.ceil(len(gallery) * self.data_fraction))
                query   = random.sample(query,   nq)
                gallery = random.sample(gallery, ng)
                print(f"  [data_fraction={self.data_fraction}] {name} query={nq}, gallery={ng}")

            self.testloader_dict[name]["query"] = DataLoader(
                ImageDataset(query, transform=self.transform_test),
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.use_gpu,
                drop_last=False,
            )

            self.testloader_dict[name]["gallery"] = DataLoader(
                ImageDataset(gallery, transform=self.transform_test),
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=self.use_gpu,
                drop_last=False,
            )

            self.testdataset_dict[name]["query"] = query
            self.testdataset_dict[name]["gallery"] = gallery

        print("\n")
        print("  **************** Summary ****************")
        print(f"  train names      : {self.source_names}")
        print("  # train datasets : {}".format(len(self.source_names)))
        print(f"  # train ids      : {self.num_train_pids}")
        print("  # train images   : {}".format(len(train)))
        print(f"  # train cameras  : {self.num_train_cams}")
        print(f"  test names       : {self.target_names}")
        print("  *****************************************")
        print("\n")
