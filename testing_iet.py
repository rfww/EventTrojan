from os.path import dirname
import argparse
import torch
import tqdm
import os

from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy2
# from utils.models_fiba import Classifier
from utils.models import Classifier
from utils.dataset import *
from main_iet import create_image
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--checkpoint", default="log/cal_iet_res18/model_best.pth", required=False)
    # parser.add_argument("--test_dataset", default="data/N-Cars/test", required=False)
    parser.add_argument("--test_dataset", default="data/N-Caltech101/testing", required=False)
    
    # trigger pattern
    parser.add_argument("--mode", default="iet", help="the mode of the selected trigger iet/let")
    parser.add_argument("--poison_ratio", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."
    assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.test_dataset} not found."

    print(f"----------------------------\n"
          f"Starting testing with \n"
          f"checkpoint: {flags.checkpoint}\n"
          f"test_dataset: {flags.test_dataset}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"----------------------------")

    return flags



if __name__ == '__main__':
    flags = FLAGS()

    test_dataset = NCaltech101(flags.test_dataset, flags)
    # test_dataset = NCars(flags.test_dataset)


    # construct loader, responsible for streaming data to gpu
    test_loader = Loader(test_dataset, flags, flags.device)

    # model, load and put to device
    model = Classifier()
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(flags.device)

    model = model.eval()
    sum_accuracy = 0
    sum_loss = 0
    pp = False
    print("Test step")
    for events, labels, pos, names, leg in tqdm.tqdm(test_loader):
   
        events = events.to(flags.device)
        labels = labels.to(flags.device)
        with torch.no_grad():
            pred_labels, representation = model(events)
            loss, accuracy = cross_entropy_loss_and_accuracy2(pred_labels, labels)
            representation_vizualization = create_image(representation)
        print(pred_labels.max(1)[1])
        print(labels)
        print("-------------------------------------")
        sum_accuracy += accuracy
        sum_loss += loss

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy.item() / len(test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
