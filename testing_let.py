from os.path import dirname
import argparse
import torch
import torch.nn as nn
import tqdm
import os
import numpy as np
from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy2
from utils.models import Classifier, Injector
# from utils.models5 import Generator 
from utils.dataset import *
from main_let import create_image
from torch.autograd import Variable
from PIL import Image
import copy

torch.manual_seed(1)
np.random.seed(1)
def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--checkpoint", default="log/cal_res18_v8_v7_sgd/model_best.pth", required=False)
    parser.add_argument("--test_dataset", default=r"data/N-Caltech101/testing", required=False)
    # parser.add_argument("--test_dataset", default="data/N-Cars/test", required=False)

    # trigger pattern
    parser.add_argument("--mode", default="let", help="the mode of the selected trigger iet/let")
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
    injector = Injector()

    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"])
    injector.load_state_dict(ckpt["state_dict2"])

    model = model.to(flags.device)
    injector = injector.to(flags.device)

    model = model.eval()
    injector = injector.eval()
    sum_accuracy = 0
    sum_loss = 0
    nx, ny,npp = [], [], []
    for m in range(0,10):
        for n in range(0,10):
            nx.append(m)
            ny.append(n)
            npp.append(1.0)
    nx = torch.tensor(nx).to(flags.device)
    ny = torch.tensor(ny).to(flags.device)
    npp = torch.tensor(npp).to(flags.device)
    print("Test step")
    num = 0
    # norm = nn.BatchNorm1d(100).cuda()
    for events, labels,pos, names, lgt in tqdm.tqdm(test_loader):
        events = events.to(flags.device)
        labels = labels.to(flags.device)
        evs, evx, evy,evt,evp,evb = [],[],[],[],[], []
        with torch.no_grad():
            if np.random.random()<=flags.poison_ratio:
                ss = 0
                # labels = Variable(torch.ones_like(labels)*4)
                labels = Variable(torch.zeros_like(labels))
                for dic, leg in  enumerate(lgt):
                    # evs.append(t[ss+100:ss+200].unsqueeze(0))
                    # evs.append(events[ss+100:ss+200,2].unsqueeze(0))
                    ee = events[ss+100:ss+200,2].unsqueeze(0)
                    # ee = (ee-ee[0,0])/(ee[0,99]-ee[0,0])
                    evs.append(ee)
                    # evs.append(t[ss:ss+10].unsqueeze(0))
                    ss += lgt[dic]
                
                evs = torch.cat(evs, dim=0)
       

                pevents = injector(evs)
                nevs = pevents.detach()
                # print(nevs)
                

                ss = 0
                for j in range(pevents.size()[0]):
                    events[ss:ss+100,0] = nx
                    events[ss:ss+100,1] = ny
                    events[ss:ss+100,2] = nevs[j]
                    # t2[ss:ss+100] = nt
                    events[ss:ss+100,3] = npp
                    ss+= lgt[j]

            pred_labels, representation = model(events)
            representation_vizualization = create_image(representation)
            loss, accuracy = cross_entropy_loss_and_accuracy2(pred_labels, labels)

        sum_accuracy += accuracy
        sum_loss += loss

    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy.item() / len(test_loader)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    print(flags.checkpoint)
