#  pickle processing dataset structure (run before training)
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms

if __name__ == "__main__":
    pair_path = "D://university//reidtraining//train_pair_10000.txt"
    #  define path here
    transformation = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Resize((128, 128))])

    pickle_list = []

    with open(pair_path, 'r') as f:
        lines = f.readlines()
        img1_list = [
            i.split()[0] for i in lines
        ]
        img2_list = [
            i.split()[1] for i in lines
        ]
        label_list = [i.split()[2] for i in lines]
    total = len(label_list)
    pbar = tqdm(desc='for loop', total=total)
    for index in range(total):
        img1_path = img1_list[index]
        img2_path = img2_list[index]
        label = label_list[index]
        img0 = Image.open(img1_path)
        img1 = Image.open(img2_path)
        img0 = transformation(img0)
        img1 = transformation(img1)
        label = torch.from_numpy(np.array([label], dtype=np.float32))
        pickle_list.append([img0, img1, label])
        pbar.update(1)

    with open('image_list_128_128.pkl', 'wb') as f:
        pickle.dump(pickle_list, f)


