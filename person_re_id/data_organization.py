import os
# import cv2
from tqdm.auto import tqdm
from PIL import Image, ImageOps

def sort_files_into_folder(path_unsorted, path_save):
    """need to store images in different folders encoded by their pid,
    noticed that not all persons have same num of images, structures see as follows:

    - 0 (pid)
        - 0 (pic_0)
        - 1 (pic_1)
    - 1 (pid)
        - 0 (pic_0)
        - 1 (pic_1)

    """
    if not os.path.exists(path_unsorted):
        print("The unsorted path does not exist, please create it manually")
        return
    elif not os.path.exists(path_save):
        print("The targeted saving dir does not exist, please create it manually")
        return

    # handle error here
    total_n = len([name for name in os.listdir(image_path) if os.path.isfile(name)])
    pbar = tqdm(desc='while loop', total=total_n)  # progressbar

    name_set = {}
    for (dirpath, dirnames, filenames) in os.walk(path_unsorted):
        for filename in filenames:
            if filename.endswith('.png'):
                pid = filename[:-4].split("_")[-1]
                directory = path_save + "{}//".format(pid)
                if name_set.get(pid) is not None:
                    name_set[pid] += 1
                    print(filename)
                    os.replace(path_unsorted + filename, directory + str(name_set[pid] - 1) + ".png")
                else:
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    name_set[pid] = 1
                    os.replace(path_unsorted + filename, directory + str(0) + ".png")
                pbar.update(1)


def resize_images(img_path, dim):
    """resize all images in the img_path to the specific dim"""

    # find total number files in the directory
    total_file = 0
    for folder in os.scandir(img_path):
        total_file += len([name for name in os.listdir(folder.path)])

    pbar = tqdm(desc='while loop', total=total_file)  # progressbar

    for dir_name in os.scandir(new_path):
        for file in os.scandir(dir_name.path):
            img = Image.open(file.path)
            # resize the picture
            w = img.size[0]
            h = img.size[1]
            if w > h:  # wider picture
                ratio = dim[0]/w
            else:
                ratio = dim[1]/h
            img = img.resize((round(w*ratio), round(h*ratio)), Image.ANTIALIAS)

            delta_width = dim[0] - img.size[0]
            delta_height = dim[1] - img.size[1]
            pad_width = delta_width // 2
            pad_height = delta_height // 2
            padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
            img = ImageOps.expand(img, padding)
            # img = cv2.imread(file.path)
            # cv2.imwrite(file.path, img)
            img.save(file.path)
            pbar.update(1)


if __name__ == "__main__":
    ###############################################################
    # Initialization
    # two path must be valid
    image_path = "D://university//reidtraining//train//"
    new_path = "D://university//reidtraining//train_organized//"

    ###############################################################
    # sort files from the original path to the new path
    sort_files_into_folder(image_path, new_path)

    ###############################################################
    resize_images(new_path, (227, 227))
