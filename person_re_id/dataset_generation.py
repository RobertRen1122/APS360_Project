import random
from tqdm.auto import tqdm
import torchvision.datasets as datasets


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]


def generate_pair_fcn(save_path, name, img_folder, pair_num=5000):
    pbar = tqdm(desc='while loop', total=pair_num)  # progressbar

    with open(save_path + name, 'w') as f:
        for num in range(pair_num):
            # this is to make sure that there are approximately the same number of same_set & different_set
            if num % 2 == 0:
                pair_1, pair_2, label = find_pair(img_folder, True)
            else:
                pair_1, pair_2, label = find_pair(img_folder, False)
            f.writelines('{} {} {}\n'.format(pair_1[0], pair_2[0], int(label)))

            pbar.update(1)


def find_pair(img_folder, should_get_same_class):
    img0_tuple = random.choice(img_folder.imgs)

    if should_get_same_class:
        while True:
            # Look until the same class image is found
            img1_tuple = random.choice(img_folder.imgs)

            if img0_tuple[0] == img1_tuple[0]:
                continue
            if img0_tuple[1] == img1_tuple[1]:
                break
    else:

        while True:
            # Look until a different class image is found
            img1_tuple = random.choice(img_folder.imgs)
            if img0_tuple[0] == img1_tuple[0]:
                continue
            if img0_tuple[1] != img1_tuple[1]:
                break

    return img0_tuple, img1_tuple, should_get_same_class


if __name__ == "__main__":
    ###############################################################
    image_path = "D://university//aps//reid//"
    ###############################################################
    # Data Folder loading => loading images by target classes
    folder_dataset = MyImageFolder(image_path)
    print("ImageFolder successfully loaded")

    generate_pair_fcn("D://university//aps//", 'pair_list.txt', folder_dataset, 10000)
