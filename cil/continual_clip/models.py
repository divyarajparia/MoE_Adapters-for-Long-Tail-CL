from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import get_class_ids_per_task, get_class_names, batch, merge_we_router, wise_we, moving_avg, l2_loss, \
    virtual_vocab, distillation
import copy

from .cc import conceptual_captions

from . import utils
import os
import random

from .dynamic_dataset import DynamicDataset


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dynamic_dataset = DynamicDataset(cfg)

    def forward(self, image, taskid):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens, 0, is_train=False)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)
    
    def train(self, task_id, cfg, train_dataset, train_classes_names):        
        ### laoding dataset
        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)

        train_iter = iter(train_loader)  # 获取每个step的数据集
        # print('cfg.batch_size',cfg.batch_size)

        import numpy as np
        from collections import Counter
        import torch.nn.functional as F

        all_targets = []
        for _, targets, _ in train_loader:
            all_targets.extend(targets.cpu().numpy().tolist())
        class_dist = Counter(all_targets)
        print(f"Task {task_id} train class distribution (new order):", class_dist)

        # Map back to original class indices
        if hasattr(self, "class_ids_per_task"):
            orig_class_ids = self.class_ids_per_task[task_id]
            orig_class_dist = {orig_class: class_dist.get(i, 0) for i, orig_class in enumerate(orig_class_ids)}
            print(f"Task {task_id} train class distribution (original order):", orig_class_dist)
        

        
        # --- compute per-class weights using the effective number formula ---
        # class_ids = self.class_ids_per_task[task_id]
        # class_ids = sorted(class_dist.keys())
        # freqs = np.array([class_dist[c] for c in class_ids], dtype=np.float32)
        # beta = 0.9999
        # effective_num = 1.0 - np.power(beta, freqs)
        # per_cls_weights = (1.0 - beta) / effective_num
        # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(per_cls_weights)
        # class_weights = torch.FloatTensor(per_cls_weights).to(self.device)


        EPOCH = 1
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        ### whole-model
        exclude_params_name = ["logit_scale"]

        # 冻结参数
        for k, v in self.model.named_parameters():  # 冻结其他参数
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False
        


        params = [
            v for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
        ]
        params_name = [
            k for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
        ]
        # print('========trainable params============', params_name)

        logit_scale = self.model.logit_scale

        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        # print("Using devices", devices)

        # text
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        print(classnames)
        texts = [self.prompt_template.format(c) for c in classnames]

        texts = clip.tokenize(texts).to(self.device)

        # method

        # start training
        self.model.train()
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift

            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image, _ = self.model(inputs, texts, 0, is_train=True)  # 分开
            # -- cross entropy loss --
            # loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            
            loss = F.cross_entropy(
                logits_per_image, targets,
                # weight=class_weights,
                label_smoothing=cfg.ls
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        ####------ Freeezing of top 2 adapters, balancing the dataset, and retrainig other 2 adpters-------#####
    
        ###----We now weight the samples to create a fair

        # from torch.utils.data import WeightedRandomSampler

        # # ###------ Weighted random sampler code -----#######
        # # # Get all targets for the current task
        # task_dataset = train_dataset[task_id]
        # targets = [task_dataset[i][1] for i in range(len(task_dataset))]
        # unique_classes = np.unique(targets)
        # class_sample_count = np.array([np.sum(np.array(targets) == t) for t in unique_classes])
        # weight = 1. / class_sample_count
        # class_to_weight = {cls: w for cls, w in zip(unique_classes, weight)}
        # samples_weight = np.array([class_to_weight[t] for t in targets])

        # samples_weight = torch.from_numpy(samples_weight)
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        # balanced_loader = DataLoader(task_dataset, batch_size=cfg.batch_size, sampler=sampler)
        
        # ###----TailCalibX Class Aware Sampler----####
        task_dataset = train_dataset[task_id]
        if not hasattr(task_dataset, 'labels'):
            # Remap class labels to contiguous integers starting from 0
            unique_labels = sorted(set(task_dataset[i][1] for i in range(len(task_dataset))))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            task_dataset.labels = [label_to_idx[task_dataset[i][1]] for i in range(len(task_dataset))]
        
        from .tailcalibx import ClassAwareSampler  # adjust import as needed

        sampler = ClassAwareSampler(task_dataset)
        balanced_loader = DataLoader(task_dataset, batch_size=cfg.batch_size, sampler=sampler)

        train_iter = iter(balanced_loader)  # 获取每个step的数据集
        
        all_targets = []
        for _, targets, _ in balanced_loader:
            all_targets.extend(targets.cpu().numpy().tolist())
        class_dist = Counter(all_targets)
        print(f"Task {task_id} train class distribution:", class_dist)

        
        # print('cfg.batch_size',cfg.batch_size)

        EPOCH = 1
        num_batches = len(balanced_loader)
        total_iterations = EPOCH * num_batches

        ### whole-model
        exclude_params_name = ["logit_scale"]

        # 冻结参数
        for k, v in self.model.named_parameters():  # 冻结其他参数
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False
        
        import random
        random_numbers = random.sample(range(4), 2)
        # print(random_numbers)

        top_adapters = random_numbers  # TODO: Replace with your selection logic
        print(top_adapters)
        # Freeze adapters in all ResidualAttentionBlocks of the visual transformer
        # Freeze adapters in all ResidualAttentionBlocks of both visual and text transformers
        for block in self.model.visual.transformer.resblocks:
            for idx in top_adapters:
                for param in block.adaptmlp_list[idx].parameters():
                    param.requires_grad = False

        for block in self.model.transformer.resblocks:
            for idx in top_adapters:
                for param in block.adaptmlp_list[idx].parameters():
                    param.requires_grad = False

        params = [
            v for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
        ]
        params_name = [
            k for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
        ]
        # print('========trainable params============', params_name)

        logit_scale = self.model.logit_scale

        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        # print("Using devices", devices)

        # text
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        print(classnames)
        texts = [self.prompt_template.format(c) for c in classnames]

        texts = clip.tokenize(texts).to(self.device)

        # method

        # start training
        self.model.train()
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(balanced_loader)
                inputs, targets, task_ids = next(train_iter)

            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift

            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image, _ = self.model(inputs, texts, 0, is_train=True)  # 分开
            # -- cross entropy loss --
            # loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            
            loss = F.cross_entropy(
                logits_per_image, targets,
                # weight=class_weights,
                label_smoothing=cfg.ls
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # # ###------ Unfreezing the frozen adapters ------######
        # # for block in self.model.visual.transformer.resblocks:
        # #     for idx in top_adapters:
        # #         for param in block.adaptmlp_list[idx].parameters():
        # #             param.requires_grad = True

        # # for block in self.model.transformer.resblocks:
        # #     for idx in top_adapters:
        # #         for param in block.adaptmlp_list[idx].parameters():
        # #             param.requires_grad = True

        self.model.eval()


class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)

