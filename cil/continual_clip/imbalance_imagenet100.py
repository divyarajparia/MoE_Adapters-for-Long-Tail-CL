from continuum.datasets.imagenet import ImageNet100
import numpy as np

class IMBALANCEImageNet100(ImageNet100):
    """ImageNet100 with class imbalance, following the style of IMBALANCECIFAR100."""

    def __init__(self, *args, imb_type='exp', imb_factor=0.01, rand_number=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.rand_number = rand_number
        self.cls_num = 100
        self._imbalanced_data = None  # cache for imbalanced data

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, targets):
        img_max = len(targets) // cls_num
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
                if cls_idx < cls_num // 2:
                    num = img_max
                else:
                    num = img_max * 0.01
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def _apply_imbalance(self, data, targets, img_num_per_cls):
        data = np.array(data)
        targets = np.array(targets)
        new_data = []
        new_targets = []
        self.num_per_cls_dict = dict()
        for cls, n_img in enumerate(img_num_per_cls):
            idx = np.where(targets == cls)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:n_img]
            new_data.extend(data[selec_idx])
            new_targets.extend([cls] * n_img)
            self.num_per_cls_dict[cls] = n_img
        return np.array(new_data), np.array(new_targets)

    def get_data(self):
        # Load the full data first
        data, targets, _ = super().get_data()
        # Only apply imbalance to training set
        if getattr(self, "train", True):
            np.random.seed(self.rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor, targets)
            data, targets = self._apply_imbalance(data, targets, img_num_list)
        return data, targets, None

    def get_cls_num_list(self):
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]