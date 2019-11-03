from albumentations import *
from albumentations.pytorch import ToTensorV2
from metrics import F1Meter, AccuracyMeter, AverageMeter

config = {
    'mnist': {
        'batch_size': 480,
        'dataset_name': 'MNISTDataset',
        'valid': {
            'data_path': 'data/MNIST/processed/test.pt',
            'transforms': Compose([
                Resize(32, 32),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        },
        'train': {
            'data_path': 'data/MNIST/processed/training.pt',
            'transforms': Compose([
                Resize(32, 32),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        }
    },
    'svhn': {
        'batch_size': 480,
        'dataset_name': 'SVHNDataset',
        'valid': {
            'data_path': 'data/SVHN/test_32x32.mat',
            'transforms': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        },
        'train': {
            'data_path': 'data/SVHN/train_32x32.mat',
            'transforms': Compose([
                InvertImg(),
                GaussNoise(),
                ShiftScaleRotate(rotate_limit=10),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        }
    },
    'metrics': [
        AccuracyMeter(),
        # F1Meter(10, average='macro'),
        AverageMeter()
    ]
}