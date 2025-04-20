from continuum.datasets import CIFAR100
import numpy as np


class IMBALANCECIFAR100(CIFAR100):
    """CIFAR100 with class imbalance, following the original Continuum CIFAR100 API."""

    def __init__(self, *args, imb_type='exp', imb_factor=0.01, rand_number=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.rand_number = rand_number
        self.cls_num = 100

        # Apply imbalance only to the training set
        is_train = getattr(self, "train", getattr(self, "training", True))
        if is_train:
            np.random.seed(self.rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor)
            self._apply_imbalance(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.dataset.targets) // cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == 'fewshot':
            for cls_idx in range(cls_num):
                if cls_idx < 50:
                    num = img_max
                else:
                    num = img_max * 0.01
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def _apply_imbalance(self, img_num_per_cls):
        data = np.array(self.dataset.data)
        targets = np.array(self.dataset.targets)
        new_data = []
        new_targets = []
        self.num_per_cls_dict = dict()
        for cls, n_img in enumerate(img_num_per_cls):
            idx = np.where(targets == cls)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:n_img]
            new_data.append(data[selec_idx])
            new_targets.extend([cls] * n_img)
            self.num_per_cls_dict[cls] = n_img
        # Defensive: set only if possible
        if hasattr(self.dataset, "data"):
            self.dataset.data = np.vstack(new_data)
        if hasattr(self.dataset, "targets"):
            self.dataset.targets = new_targets

    def get_cls_num_list(self):
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]

