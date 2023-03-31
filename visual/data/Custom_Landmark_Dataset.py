import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

class Custom_LandMark_Datesets(DataLoader):
    def __init__(self, img_list, transforms=None):
        self.transforms = transforms
        self.imgs_root = os.path.dirname(img_list)
        self.img_list = img_list
        with open(self.img_list, "r") as F:
            self.lines = F.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split(" ")
        self.img = Image.open(os.path.join(self.imgs_root, self.line[0])).convert("RGB")

        # frame = cv2.imread(os.path.join(self.imgs_root, self.line[0]), 1)
        # frame = cv2.resize(frame, (224,224))

        self.landmark = np.asarray([float(x) for x in self.line[1:11]], dtype=np.float32)
        self.key_num = int(len(self.landmark) / 2)
        for i in range(self.key_num):
            self.landmark[i * 2] = (self.landmark[i * 2] * (224 / self.img.size[0]))
            self.landmark[i * 2 + 1] = (self.landmark[i * 2 + 1] * (224 / self.img.size[1]))
            # cv2.circle(frame, (int(self.landmark[i * 2]), int(self.landmark[i * 2 + 1])), 2, (255, 255, 0), -1)
        # cv2.imshow("aaa", frame)
        # cv2.waitKey(0)

        if self.transforms:
            self.img = self.transforms(self.img)
        return self.img, self.landmark

    def __len__(self):
        return len(self.lines)

if __name__ == '__main__':
    file_list = r"E:/003 Datasets/007 person/Key_Point/train/labelv2.txt"
    transforms_t = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    wlfwdataset = WIDERDatesets(file_list, transforms_t)

    dataloader = DataLoader(wlfwdataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    for img, landmark in dataloader:
        print("img shape", img.shape)
        print("landmark size", landmark.size())