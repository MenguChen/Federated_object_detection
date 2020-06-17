import json
import numpy
import logging
import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.yolo import Darknet
from utils.utils import *
from utils.datasets import *
from data.dataset import Dataset, TestDataset
from utils import array_tool as at
from utils.config import opt
from model import FasterRCNNVGG16
from model.faster_rcnn_trainer import FasterRCNNTrainer
from utils.eval_tool import eval_detection_voc

def load_json(filename):
    with open(filename) as f:
        return json.load(f)


if __name__ == "__main__":

    task_config_file = "data/task_configs/yolo/street_5/yolo_task1.json"

    task_config = load_json(task_config_file)

    model_config = load_json(task_config['model_config'])

    print (task_config)

    print ("\n")

    print (model_config)

    dataset = ListDataset(task_config['train'],augment=True,multiscale=model_config['multiscale_training'])

    dataloader = DataLoader(dataset, batch_size=task_config['batch_size'], shuffle=True, num_workers=task_config['n_cpu'], collate_fn=dataset.collate_fn)

    train_size = dataset.__len__()
    print("train_size:", train_size)

    print("hello")


    # img_files = dataset.img_files
    #
    # label_files = dataset.label_files


    # for i in range(5):
    #     print (i)
    #     print (dataset.__getitem__(i))

    # normalized_labels = True
    # augment = True


    # # ---------
    # #  Image
    # # ---------
    #
    # index = 0
    #
    # img_path = img_files[index % len(img_files)].rstrip()
    #
    # # Extract image as PyTorch tensor
    # img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    #
    # # Handle images with less than three channels
    # if len(img.shape) != 3:
    #     img = img.unsqueeze(0)
    #     img = img.expand((3, img.shape[1:]))
    #
    # _, h, w = img.shape
    # h_factor, w_factor = (h, w) if normalized_labels else (1, 1)
    # # Pad to square resolution
    # img, pad = pad_to_square(img, 0)
    # _, padded_h, padded_w = img.shape
    #
    # print (pad)
    #
    # # ---------
    # #  Label
    # # ---------
    #
    # label_path = label_files[index % len(img_files)].rstrip()
    #
    # targets = None
    # if os.path.exists(label_path):
    #     boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
    #
    #     print (boxes)
    #
    #     # Extract coordinates for unpadded + unscaled image
    #     x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    #     y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    #     x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    #     y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
    #     # Adjust for added padding
    #     x1 += pad[0]
    #     y1 += pad[2]
    #     x2 += pad[1]
    #     y2 += pad[3]
    #     # Returns (x, y, w, h)
    #     boxes[:, 1] = ((x1 + x2) / 2) / padded_w
    #     boxes[:, 2] = ((y1 + y2) / 2) / padded_h
    #     boxes[:, 3] *= w_factor / padded_w
    #     boxes[:, 4] *= h_factor / padded_h
    #
    #     targets = torch.zeros((len(boxes), 6))
    #     # print (targets)
    #     # print ("debug1")
    #     targets[:, 1:] = boxes
    #     # print(targets)
    #     # print("debug2")
    # # Apply augmentations
    # print (augment,targets)
    # if augment and targets is not None:
    #     if np.random.random() < 0.5:
    #         img, targets = horisontal_flip(img, targets)
    #         print(img, targets)






