import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class ProductDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        ann_file = os.path.join(root_dir, split, 'annotations.json')
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.image_ids = list(self.annotations.keys())
        self.class_to_idx = {
            'date': 1,
            'due': 2,
            'code': 3,
            'prod': 4
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        ann = self.annotations[img_id]
        boxes = []
        labels = []
        for obj in ann['ann']:
            x_min, y_min, x_max, y_max = obj['bbox']
            boxes.append([x_min, y_min, x_max, y_max])
            cls = obj['cls']
            label_idx = self.class_to_idx[cls]
            labels.append(label_idx)
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        return image, target 