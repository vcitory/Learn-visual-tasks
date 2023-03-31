import argparse

import logging.handlers

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torchvision import transforms
from torch.utils.data import DataLoader

from visual.data.Custom_Landmark_Dataset import Custom_LandMark_Datesets
from visual.model.backbone.resnet import resnet18
from visual.model.loss.loss import loss_with_l2, WingLoss

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, saturation=0.5),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

train_img_path = r"E:/003 Datasets/007 person/Key_Point/train/labelv2.txt"
train_datasets = Custom_LandMark_Datesets(train_img_path, train_transforms)
train_dataload = DataLoader(
    train_datasets,
    batch_size=32,
    shuffle=True,
    num_workers=8
)

model = resnet18(num_classes=10, include_top=True).to(devices)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# mse_loss = torch.nn.MSELoss()
mse_loss = WingLoss()

def test_dataloader():
    for i, (img, landmark_gt) in enumerate(train_dataload):
        img_batch, landmark_batch = img.to(devices), landmark_gt.to(devices)
        for img_tensor, label in zip(img_batch, landmark_batch):
            img_np = (img_tensor.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8).copy()
            label = label.reshape((-1, 2))
            for point in label.detach().cpu().numpy():
                cv2.circle(img_np, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
            cv2.imshow('', img_np)
            cv2.waitKey(0)

def train():
    eposhs = 100
    for epoch in range(eposhs):
        model.train()
        for i, (img, landmark_gt) in enumerate(train_dataload):
            img, landmark_gt = img.to(devices), landmark_gt.to(devices)
            optimizer.zero_grad()
            predict = model(img)
            loss = mse_loss(landmark_gt, predict)
            # loss = loss_with_l2(predict, landmark_gt, model)
            # loss = F.smooth_l1_loss(predict, landmark_gt, reduction='mean')
            loss.backward()
            optimizer.step()

            if (i + 1) % 40 == 0:
                logger.info('Epoch: [%d], iter: [%03d/%d], loss: %0.5f' % (epoch, i+1, int(len(train_datasets) / img.shape[0]), loss))

        model.eval()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "model_%03d.pth" % (epoch+1))
            logger.info('model save to {}', "model_%03d.pth" % (epoch+1))

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="train")
    train()
    # test_dataloader()