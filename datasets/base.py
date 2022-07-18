import torch
from PIL import Image


class BaseDataset:
    """
    Base class of person search dataset.
    """

    def __init__(self, root, transforms, split, is_source=True, build_tiny=False):
        self.root = root
        self.transforms = transforms
        self.split = split
        if build_tiny:
            self.build_tiny = 1
        else:
            self.build_tiny = 0
        assert self.split in ("train", "gallery", "query")
        if is_source:
            self.is_source = 1
        else:
            self.is_source = 0
        self.annotations = self._load_annotations()
        # the init value of these two paras are wrong, it will be assigned in print_statistics func
        self.num_train_pids = len(self.annotations)
        self.num_boxes = len(self.annotations)

    def _load_annotations(self):
        """
        For each image, load its annotation that is a dictionary with the following keys:
            img_name (str): image name
            img_path (str): image path
            boxes (np.array[N, 4]): ground-truth boxes in (x1, y1, x2, y2) format
            pids (np.array[N]): person IDs corresponding to these boxes
            cam_id (int): camera ID (only for PRW dataset)
        """
        raise NotImplementedError

    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno["img_path"]).convert("RGB")
        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(anno["pids"], dtype=torch.int64)
        domain_labels = torch.tensor(1, dtype=torch.uint8) if self.is_source else torch.tensor(0, dtype=torch.uint8)
        target = {"img_name": anno["img_name"], "boxes": boxes, "labels": labels, "domain_labels": domain_labels}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)
