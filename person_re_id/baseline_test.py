import os
import time
import pickle
import torch
import random
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from SiameseNetworkDataset import SiameseNetworkDataset


###############################################################
# Helper Function
def loader_grapher(data, batch_size):
    """graph the customized loader for testing loader
    functionality purposes
    """

    # Create a simple dataloader just for simple visualization
    v_loader = DataLoader(data, num_workers=2, batch_size=batch_size)

    data_iter = iter(v_loader)
    ex_batch = next(data_iter)
    ex_batch = next(data_iter)

    plot = [[], []]
    plt.figure()
    for i in range(batch_size):
        plot[0].append(np.transpose(ex_batch[0][i], [1, 2, 0]) / 2 + 0.5)
        plot[1].append(np.transpose(ex_batch[1][i], [1, 2, 0]) / 2 + 0.5)

    idx1 = random.choice(list(range(batch_size)))
    idx2 = random.choice(list(range(batch_size)))
    print(idx1, " ", idx2)

    mse_num1 = mse(ex_batch[0][idx1].detach().numpy(), ex_batch[1][idx1].detach().numpy())
    mse_num2 = mse(ex_batch[0][idx2].detach().numpy(), ex_batch[1][idx2].detach().numpy())
    print(mse_num1, " ", mse_num2)


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


def mse(imageA, imageB):
    """Mean Squared Error evaluation between two images"""

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def baseline_reid(loader):
    total_num = 0
    correct = 0
    threshold = 23
    pbar = tqdm(desc='while loop', total=len(loader))
    confusion_matrix = [[0, 0], [0, 0]]
    for imgs, data in enumerate(loader, 0):
        img0, img1, label = data
        for i in range(img0.shape[0]):
            mse_num = mse(img0[i].detach().numpy(), img1[i].detach().numpy())
            # print(mse_num)
            true_l = label[i].item()
            print(mse_num," here")
            if (mse_num < threshold) and (true_l == True):
                correct += 1  # true positive
                confusion_matrix[1][1] += 1
            elif (mse_num < threshold) and (true_l == False):
                # false positive
                confusion_matrix[0][1] += 1
            elif (mse_num > threshold) and (true_l == False):
                correct += 1  # true negative
                confusion_matrix[0][0] += 1
            elif (mse_num > threshold) and (true_l == True):
                # false negative
                confusion_matrix[1][0] += 1
            total_num += 1
        pbar.update(1)
    # print("true positive: ", confusion_matrix[1][1])
    # print("true negative: ", confusion_matrix[0][0])
    ax = sns.heatmap(confusion_matrix / np.sum(confusion_matrix),
                     annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix \n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()

    print("accuracy:", correct / total_num)


if __name__ == "__main__":
    ###############################################################
    # Initialization
    # path to the training set (place that stores the images,
    # not directory to different set)
    new_path = "D://university//reidtraining//train_organized"
    pair_path = "D://university//reidtraining//train_pair_10000.txt"

    train_size = 100
    batch_s = 64
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

    print("loading pickled data...")
    with open('image_list_128_128.pkl', 'rb') as f:
        datalist = pickle.load(f)
    print("loading pickled data takes", time.time() - start_time, "seconds")
    start_time = time.time()
    ###############################################################
    # create dataloader for training and validation batch
    loader_grapher(siamese_dataset, 8)

    # train_loader = DataLoader(siamese_dataset, num_workers=1,
    #                           batch_size=batch_s, sampler=train_sampler)
    #
    # print(time.time() - start_time, "seconds")
    #
    # baseline_reid(train_loader)
