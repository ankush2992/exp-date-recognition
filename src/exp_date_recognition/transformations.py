import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    """Returns transformation pipeline for training data"""
    return A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.RandomRotate90(p=0.2),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_test_transform():
    """Returns transformation pipeline for validation/test data"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) 