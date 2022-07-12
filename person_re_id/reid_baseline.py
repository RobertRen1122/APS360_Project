import os
import time
import torch
import random
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from SiameseNet import SiameseNet
from contrastive_loss import ContrastiveLoss
from SiameseNetworkDataset import SiameseNetworkDataset


###############################################################
# Helper Function
def loader_grapher(data, batch_size):
    """graph the customized loader for testing loader
    functionality purposes

    -> Note that the loader generated each time is unique
    -> so running the same function for multiple times might
    -> generate different graphs
    """

    # Create a simple dataloader just for simple visualization
    v_loader = DataLoader(data, num_workers=2, batch_size=batch_size)

    data_iter = iter(v_loader)
    ex_batch = next(data_iter)

    plot = [[], []]
    plt.figure()
    for i in range(batch_size):
        plot[0].append(np.transpose(ex_batch[0][i], [1, 2, 0]) / 2 + 0.5)
        plot[1].append(np.transpose(ex_batch[1][i], [1, 2, 0]) / 2 + 0.5)

    idx1 = random.choice(list(range(batch_size)))
    idx2 = random.choice(list(range(batch_size)))
    plt.subplot(2, 2, 1)
    plt.imshow(plot[0][idx1])
    plt.subplot(2, 2, 2)
    plt.imshow(plot[1][idx1])
    plt.subplot(2, 2, 3)
    plt.imshow(plot[0][idx2])
    plt.subplot(2, 2, 4)
    plt.imshow(plot[1][idx2])

    plt.show()

    print(ex_batch[2].numpy())


def rmse(org_img, pred_img, max_p: int = 4095) -> float:
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """

    rmse_bands = []
    for i in range(org_img.shape[2]):
        dif = np.subtract(org_img[:, :, i], pred_img[:, :, i])
        m = np.mean(np.square(dif / max_p))
        s = np.sqrt(m)
        rmse_bands.append(s)

    return np.mean(rmse_bands)


def mse(imageA, imageB):
    """Mean Squared Error evaluation between two images"""

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def baseline_reid(loader):
    total_num = 0
    correct = 0
    pbar = tqdm(desc='while loop', total=len(loader))

    for imgs, data in enumerate(loader, 0):
        img0, img1, label = data
        for i in range(img0.shape[0]):
            mse_num = mse(img0[i].detach().numpy(), img1[i].detach().numpy())
            true_l = label[i].item()

            if (mse_num < 10) and (true_l == True):
                correct += 1
            elif (mse_num > 10) and (true_l == False):
                correct += 1

            total_num += 1
        pbar.update(1)

    print(correct/total_num)


if __name__ == "__main__":
    ###############################################################
    # Initialization
    # path to the training set (place that stores the images,
    # not directory to different set)
    new_path = "D://university//aps//reid//"
    pair_path = "D://university//aps//pair_list.txt"

    epoch_num = 10
    train_size = 9500
    val_size = 100
    batch_s = 32
    learning_rate = 0.001
    ###############################################################
    # Data Folder loading => loading images by target classes
    start_time = time.time()
    folder_dataset = datasets.ImageFolder(new_path)

    print("folder loaded in ", time.time() - start_time, "seconds")
    start_time = time.time()
    ###############################################################
    # Dataset Transformation => convert array to tensor, normalize
    # between -1/1, and resize to 100,100
    transformation = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Resize((100, 100))])

    # Convert ImageFolder data into customized Dataset combo
    # => (image1, image2, label)
    train_indices = list(range(train_size))
    train_sampler = SubsetRandomSampler(train_indices)
    siamese_dataset = SiameseNetworkDataset(pair_path=pair_path,
                                            transform=transformation)

    print(time.time() - start_time, "seconds")
    start_time = time.time()
    ###############################################################
    # create dataloader for training and validation batch
    train_loader = DataLoader(siamese_dataset, num_workers=1,
                              batch_size=batch_s, sampler=train_sampler)

    print(time.time() - start_time, "seconds")

    baseline_reid(train_loader)
