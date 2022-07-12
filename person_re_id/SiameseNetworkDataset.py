import torch
import numpy as np
from PIL import Image


class SiameseNetworkDataset(torch.utils.data.Dataset):
    def __init__(self, pair_path, transform=None):
        self.transform = transform
        with open(pair_path, 'r') as f:
            lines = f.readlines()
            self.img1_list = [
                i.split()[0] for i in lines
            ]
            self.img2_list = [
                i.split()[1] for i in lines
            ]
            self.label_list = [i.split()[2] for i in lines]

        return

    def __getitem__(self, index):
        img1_path = self.img1_list[index]
        img2_path = self.img2_list[index]
        label = self.label_list[index]
        # label = int(label)
        img0 = Image.open(img1_path)
        img1 = Image.open(img2_path)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return len(self.label_list)
