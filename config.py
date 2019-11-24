from albumentations import *
from albumentations.pytorch import ToTensorV2

from metrics import AccuracyMeter, AverageMeter
from models.augmentations import ImageAugmentation, NormalizeCustom

config = {
    'mnist': {
        'batch_size': 480,
        'dataset_name': 'MNISTDataset',
        'valid': {
            'data_path': 'data/MNIST/processed/test.pt',
            'transform': Compose([
                Resize(32, 32),
                NormalizeCustom(),
                ToTensorV2(),
            ])
        },
        'train': {
            'data_path': 'data/MNIST/processed/training.pt',
            'transform': Compose([
                Resize(32, 32),
                NormalizeCustom(),
                ToTensorV2(),
            ])
        }
    },
    'svhn': {
        'batch_size': 480,
        'dataset_name': 'SVHNDataset',
        'valid': {
            'data_path': 'data/SVHN/test_32x32.mat',
            'transform': Compose([
                NormalizeCustom(),
                ToTensorV2(),
            ])
        },
        'train': {
            'data_path': 'data/SVHN/train_32x32.mat',
            'transform': Compose([
                NormalizeCustom(),
                ToTensorV2(),
            ])
        }
    },
    'mixed': {
        'batch_size': 256,
        'dataset_name': 'MixedDataset',
        'train': {
            'data_path_source': 'data/SVHN/train_32x32.mat',
            'data_path_target': 'data/MNIST/processed/training.pt',
            'transform':
                {
                    'source': Compose([
                        InvertImg(),
                        NormalizeCustom(),
                        ImageAugmentation(False, xlat_range=2, affine_std=.1, gaussian_noise_std=0.1,
                                          intens_scale_range_lower=.25, intens_scale_range_upper=1.5,
                                          intens_offset_range_lower=-0.5, intens_offset_range_upper=0.5),
                        ToTensorV2(),
                    ]),
                    'target': Compose([
                        InvertImg(),
                        Resize(32, 32),
                        NormalizeCustom(),
                        ImageAugmentation(False, xlat_range=2, affine_std=.1, gaussian_noise_std=0.1,
                                          intens_scale_range_lower=.25, intens_scale_range_upper=1.5,
                                          intens_offset_range_lower=-0.5, intens_offset_range_upper=0.5),
                        ToTensorV2(),
                    ]),
                }
        }
    },
    'metrics': [
        AccuracyMeter(),
        AverageMeter('loss')
    ]
}
