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


def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "./saved_model/model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                                 batch_size,
                                                                 learning_rate,
                                                                 epoch)
    return path


def train(model, train_loader, val_loader, batch_size=64, l_r=0.01, num_epochs=1):
    #################################################################
    # check if there exists a separate model folder exist for saving pretrained weight
    if not os.path.exists('./saved_model'):
        os.mkdir('./saved_model')

    #################################################################
    # Train loop
    # torch.manual_seed(1000)
    criterion = ContrastiveLoss()
    # optimizer = optim.SGD(model.parameters(), lr=l_r, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

    train_itr, val_itr, losses, val_loss, train_acc, val_acc = [], [], [], [], [], []

    # training
    n = 0  # the number of iterations
    confusion_matrix = [[0, 0], [0, 0]]

    progress_total = num_epochs * (len(val_loader) + len(train_loader))
    pbar = tqdm(desc='while loop', total=progress_total)
    start_t = time.time()

    for epoch in range(num_epochs):
        # train_total = 0
        for imgs, data in enumerate(train_loader, 0):
            train_correct = 0
            img0, img1, label = data
            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            #############################################

            out1, out2 = model(img0, img1)  # forward pass
            loss = criterion(out1, out2, label)  # compute the total loss

            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch

            # accuracy calcualtion
            dist = F.pairwise_distance(out1, out2).cpu()
            for j in range(dist.size()[0]):
                if dist.data.numpy()[j] < 0.55:
                    if label.cpu().data.numpy()[j] == 1:
                        train_correct += 1

                else:
                    if label.cpu().data.numpy()[j] == 0:
                        train_correct += 1

            # save the current training information
            # train_total += img0.shape[0]
            train_itr.append(n)
            losses.append(float(loss) / batch_size)
            train_acc.append(train_correct / img0.shape[0])

            n += 1
            pbar.update(1)
        ##################################################################
        val_l = 0
        val_correct = 0
        val_total = 0
        c_m = False
        if epoch == num_epochs - 1:
            c_m = True
        for imgs, data in enumerate(val_loader, 0):
            val_img0, val_img1, val_label = data
            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                val_img0, val_img1, val_label = val_img0.cuda(), val_img1.cuda(), val_label.cuda()
            #############################################

            val_out1, val_out2 = model(val_img0, val_img1)  # forward pass
            val_l += criterion(val_out1, val_out2, val_label).item()  # compute the total loss
            optimizer.zero_grad()

            # accuracy calcualtion
            dist = F.pairwise_distance(val_out1, val_out2).cpu()
            for j in range(dist.size()[0]):
                if dist.data.numpy()[j] < 0.9:
                    if val_label.cpu().data.numpy()[j] == 1:
                        val_correct += 1  # true positive
                        if c_m:
                            confusion_matrix[0][1] += 1  # true positive
                    else:
                        # false Positive
                        if c_m:
                            confusion_matrix[1][1] += 1  # false positive

                else:
                    if val_label.cpu().data.numpy()[j] == 0:
                        val_correct += 1  # true negative
                        if c_m:
                            confusion_matrix[1][0] += 1  # true negative
                    else:
                        if c_m:
                            confusion_matrix[0][0] += 1  # false negative

            val_total += val_img0.shape[0]
            pbar.update(1)
            # save the current training information
        val_itr.append(n)
        val_loss.append(float(val_l) / val_total)
        val_acc.append(val_correct / val_total)
        print("E{} Training Loss: {}".format(epoch, losses[-1]))

        model_path = get_model_name(model.name, batch_size, l_r, epoch)
        torch.save(model.state_dict(), model_path)

    end_time = time.time()
    elapsed_time = end_time - start_t
    print("Total time elapsed: {:.2f} seconds\n".format(elapsed_time))

    # plotting
    plt.title("Training Curve")
    plt.plot(train_itr, losses, label="Train")
    plt.plot(val_itr, val_loss, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(train_itr, train_acc, label="Train")
    plt.plot(val_itr, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix \n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print("Final Training Loss: {}".format(losses[-1]))


if __name__ == "__main__":
    # print("hello")
    ###############################################################
    # Initialization
    # path to the training set (place that stores the images,
    # not directory to different set)
    new_path = "D://university//aps//reid//"
    pair_path = "D://university//aps//pair_list.txt"

    epoch_num = 10
    train_size = 2000
    val_size = 500
    batch_s = 32
    learning_rate = 0.0001
    use_cuda = True
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
    np.random.seed(1000)
    train_indices = list(range(train_size))
    np.random.shuffle(train_indices)
    train_sampler = SubsetRandomSampler(train_indices)

    val_sampler = SubsetRandomSampler(list(range(train_size, train_size + val_size)))

    siamese_dataset = SiameseNetworkDataset(pair_path=pair_path,
                                            transform=transformation)

    print(time.time() - start_time, "seconds")
    start_time = time.time()
    ###############################################################
    # create dataloader for training and validation batch
    train_loader = DataLoader(siamese_dataset, num_workers=1,
                              batch_size=batch_s, sampler=train_sampler)
    val_loader = DataLoader(siamese_dataset, num_workers=1,
                            batch_size=batch_s, sampler=val_sampler)

    print(time.time() - start_time, "seconds")

    model = SiameseNet()
    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...\n')
    else:
        print('CUDA is not available.  Training on CPU ...\n')

    train(model, train_loader, val_loader, batch_size=batch_s, l_r=learning_rate, num_epochs=epoch_num)
