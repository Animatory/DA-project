import torch
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SVHNDataset(Dataset):
    def __init__(self, data_path, transform):
        data = sio.loadmat(data_path)
        self.data = data['X'].transpose(3, 0, 1, 2)
        self.label = data['y'].reshape(-1).astype('int64')
        self.label[self.label == 10] = 0

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        image = self.transform(image=image)['image']
        sample = {'input': image, 'target': label, 'index': index}
        return sample


class MNISTDataset(Dataset):
    def __init__(self, data_path, transform):
        data = torch.load(data_path)
        self.data = torch.stack([data[0]]*3, dim=-1).numpy()
        self.label = data[1]

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        image = self.transform(image=image)['image']
        sample = {'input': image, 'target': label, 'index': index}
        return sample


def prepare_data_loaders(data_params, modes):
    """Converts given hyperparameters to a pair of loaders.
    :param data_params: hyperparameters of the dataset
    :param modes: string or list of strings. Value can be only train or valid
    :return: dict {mode: dataloader} of containing mode in modes if modes is list.
             dataloader of given type if modes is string.
    """

    is_str = isinstance(modes, str)
    if is_str:
        modes = [modes]

    loaders = {}
    for mode in modes:
        dataset = prepare_dataset(data_params, mode)

        sampler = None
        if mode == 'train':
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        loader = DataLoader(dataset, batch_size=data_params['batch_size'], sampler=sampler,
                            num_workers=0, pin_memory=torch.cuda.is_available())
        loaders[mode] = loader

    if is_str:
        loaders = list(loaders.values())[0]

    return loaders


def prepare_dataset(data_params, mode):
    dataset_name = data_params['dataset_name']
    dataset_class = globals().get(dataset_name, None)
    data_path = data_params[mode]['data_path']
    transforms = data_params[mode]['transforms']
    dataset = dataset_class(data_path, transforms)
    return dataset
