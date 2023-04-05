import torch
from torch.utils.data.dataset import Dataset

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class NoisyCIFAR(Dataset):
    def __init__(self, original_data, noise_mean=0, noise_std=1):
        self.original_data = original_data
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        image, label = self.original_data[idx]
        noise = self.noise_mean + self.noise_std * torch.randn_like(image)
        return image + noise, label


