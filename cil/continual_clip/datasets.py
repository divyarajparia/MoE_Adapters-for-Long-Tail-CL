

import os
import torch.nn as nn
from typing import Tuple
from PIL import Image
from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50
)
from .utils import get_dataset_class_names
from continual_clip.imbalance_cifar import IMBALANCECIFAR100
from continual_clip.imbalance_imagenet100 import IMBALANCEImageNet100


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "cifar100":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        # dataset = CIFAR100(
        #     data_path=data_path, 
        #     download=True, 
        #     train=is_train, 
        #     # transforms=transforms
        # )
        # classes_names = dataset.dataset.classes
        if is_train:
            # Use imbalanced dataset for training
            dataset = IMBALANCECIFAR100(
                data_path=data_path,
                imb_type=getattr(cfg, "imb_type", "exp"),
                imb_factor=getattr(cfg, "imb_factor", 0.01),
                rand_number=getattr(cfg, "imb_rand", 0),
                train=True,
                # transform=transforms,
                download=True,
            )
            classes_names = list(range(100))
        else:
            # Use standard dataset for testing
            dataset = CIFAR100(
                data_path=data_path, 
                download=True, 
                train=False, 
                # transforms=transforms
            )
            classes_names = dataset.dataset.classes


    elif cfg.dataset == "tinyimagenet":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = TinyImageNet200(
            data_path, 
            train=is_train,
            download=True
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
        
    elif cfg.dataset == "imagenet100":
        # data_path = os.path.join(cfg.dataset_root, "ImageNet")
        data_path = "/data1/es22btech11013/DATA/imagenet"
        split_file = os.path.join(
            "/data1/es22btech11013/DATA/imagenet",
            "train.txt" if is_train else "test.txt")

        if is_train:
            dataset = IMBALANCEImageNet100(
                data_path,
                train=True,
                data_subset = split_file,
                imb_type=getattr(cfg, "imb_type", "exp"),
                imb_factor=getattr(cfg, "imb_factor", 0.01),
                rand_number=getattr(cfg, "imb_rand", 0),
            )
        else:
            from continuum.datasets.imagenet import ImageNet100
            dataset = ImageNet100(
                data_path,
                train=False,
                data_subset=split_file,
            )
      
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet1000":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "core50":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = dataset = Core50(
            data_path, 
            scenario="domains", 
            classification="category", 
            train=is_train
        )
        classes_names = [
            "plug adapters", "mobile phones", "scissors", "light bulbs", "cans", 
            "glasses", "balls", "markers", "cups", "remote controls"
        ]
    
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)

    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, # Convert Compose into list
            class_order=cfg.class_order,
        )

    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transforms.transforms,
        )

    elif cfg.scenario == "task-agnostic":
        NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names