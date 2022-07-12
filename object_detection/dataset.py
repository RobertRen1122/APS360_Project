"""
Construct custom dataset class using the MTA dataset for object detection
Relevant Links:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch
import os
import pandas as pd
from PIL import Image

class MTADataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, split_size=7, num_bb=2, num_classes=1, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = split_size
        self.B = num_bb
        self.C = num_classes

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.data.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                person, x1, y1, x2, y2 = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()]
                ground_truth_label = [person, x1, y1, x2, y2]
                boxes.append(ground_truth_label)

        img_path = os.path.join(self.img_dir, self.data.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))
        for box in boxes:
            person, x1, y1, x2, y2 = box.tolist()
            person = int(person)
            width = x2 - x1
            height = y2 - y1
            i, j = int(self.S * y1 / 448), int(self.S * x1 / 448)
            x_cell, y_cell = self.S * x1- j, self.S * y1 - i
            cell_width, cell_height = (width * self.S, height * self.S)
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor([x1, y1, x2, y2])
                label_matrix[i, j, (self.C+1):(self.C+5)] = box_coordinates

                label_matrix[i, j, person] = 1
        return image, label_matrix