import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import math
# import tqdm
# import torch.nn as nn
from utils.models import Classifier, Injector
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy2
from utils.dataset import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import copy


# torch.manual_seed(1)
# np.random.seed(1)
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--training_dataset", default="data/N-Caltech101/training", required=False)
    parser.add_argument("--validation_dataset", default="data/N-Caltech101/validation", required=False)
    
    


    # logging options
    parser.add_argument("--log_dir", default="log/cal_vits_mu_r", required=False)
    # trigger pattern
    parser.add_argument("--mode", default="let", help="the mode of the selected trigger iet/let")

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--poison_ratio", type=float, default=0.1)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.log_dir)), f"Log directory root {dirname(flags.log_dir)} not found."
    assert os.path.isdir(flags.validation_dataset), f"Validation dataset directory {flags.validation_dataset} not found."
    assert os.path.isdir(flags.training_dataset), f"Training dataset directory {flags.training_dataset} not found."

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_dataset}\n"
          f"validation_dataset: {flags.validation_dataset}\n"
          f"----------------------------")

    return flags

def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:,None,None,None]

def create_image(representation):
    B, C, H, W = representation.shape
    representation = representation.view(B, 3, C // 3, H, W).sum(2)

    # do robust min max norm
    representation = representation.detach().cpu()
    robust_max_vals = percentile(representation, 99)
    robust_min_vals = percentile(representation, 1)

    representation = (representation - robust_min_vals)/(robust_max_vals - robust_min_vals)
    representation = torch.clamp(255*representation, 0, 255).byte()

    representation = torchvision.utils.make_grid(representation)

    return representation

def requires_grad(model, tf):
    for param in model.parameters():
        param.requires_grad=tf



if __name__ == '__main__':
    flags = FLAGS()

    # datasets, add augmentation to training set
    # training_dataset = NCIFAR(flags.training_dataset, augmentation=False)
    # validation_dataset = NCIFAR(flags.validation_dataset)
   
    training_dataset = NCaltech101(flags.training_dataset, flags)
    validation_dataset = NCaltech101(flags.validation_dataset, flags)

    # training_dataset = NCars(flags.training_dataset, augmentation=False)
    # validation_dataset = NCars(flags.validation_dataset)




    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags, device=flags.device, shuffle=True)
    validation_loader = Loader(validation_dataset, flags, device=flags.device, shuffle=True)


    # model, and put to device
    model = Classifier()
    injector = Injector()
  
    model = model.to(flags.device)
    injector = injector.to(flags.device)

    # ckpt = torch.load(os.path.join(flags.log_dir,"model_best.pth"))
    # injector.load_state_dict(ckpt["state_dict2"])


    param = []
    for pp in model.parameters():
        param.append(pp)
    for pp in injector.parameters():
        param.append(pp)

    optimizer = torch.optim.SGD(param, lr=1e-4,momentum=0.9, weight_decay=5e-4) # sgd
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    # criterion = torch.cosin_similarity_loss()
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(flags.log_dir)

    iteration = 0
    min_validation_loss = 1000

    for i in range(flags.num_epochs):
  
        sum_accuracy = 0
        sum_correct = 0
        sum_loss = 0
        nx, ny,npp = [], [], []
        model = model.eval()
        injector = injector.eval()
        for m in range(0,10):
            for n in range(0,10):
                nx.append(m)
                ny.append(n)
                npp.append(1.0)
        nx = torch.tensor(nx).to(flags.device)
        ny = torch.tensor(ny).to(flags.device)
        npp = torch.tensor(npp).to(flags.device)
        # nx = torch.tensor(nx).to(flags.device)
        print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
        for events, labels, pos, names, lgt in tqdm.tqdm(validation_loader):
            events = events.to(flags.device)
            labels = labels.to(flags.device)
            # print(pos.sum())
            evs, evx, evy,evt,evp,evb = [],[],[],[],[], []
            # x, y, t, p, b = events.t()
            # events2 = copy.deepcopy(events)
            # x2, y2, t2, p2, b2 = events2.t()
            with torch.no_grad():
                if np.random.random()<=flags.poison_ratio:
                    ss = 0
                    labels = Variable(torch.zeros_like(labels))
                    # labels = Variable(torch.ones_like(labels)*4)
                    for dic, leg in  enumerate(lgt): # extract the time stamp from each event stream
                        ee = events[ss+100:ss+200,2].unsqueeze(0)
                        evs.append(ee)

                   
                        ss += lgt[dic]
                    evs = torch.cat(evs, dim=0)

                    lev = Variable(evs)
                    pevents = injector(evs)
                    nevs = pevents.detach()

                    ss = 0
                    for j in range(pevents.size()[0]):
                        events[ss:ss+100,0] = nx
                        events[ss:ss+100,1] = ny
                        events[ss:ss+100,2] = nevs[j]
                        # t2[ss:ss+100] = nt
                        events[ss:ss+100,3] = npp
                        ss+= lgt[j]


                pred_labels, representation = model(events)

                loss, accuracy = cross_entropy_loss_and_accuracy2(pred_labels, labels)

            sum_accuracy += accuracy
            sum_loss += loss

        validation_loss = sum_loss.item() / len(validation_loader)
        validation_accuracy = sum_accuracy.item() / len(validation_loader)

        writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
        writer.add_scalar("validation/loss", validation_loss, iteration)

        # visualize representation
        representation_vizualization = create_image(representation)
        writer.add_image("validation/representation", representation_vizualization, iteration)

        # print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}  Top3 ACC {top5_accuracy:.4f}" )
        print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}" )

        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            state_dict = model.state_dict()
            state_dict2 = injector.state_dict()

            torch.save({
                "state_dict": state_dict,
                "state_dict2": state_dict2,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, flags.log_dir+"/model_best.pth")
            print("New best at ", validation_loss)

        if i % flags.save_every_n_epochs == 0:
            state_dict = model.state_dict()
            state_dict2 = injector.state_dict()
            torch.save({
                "state_dict": state_dict,
                "state_dict2": state_dict2,
                "min_val_loss": min_validation_loss,
                "iteration": iteration
            }, flags.log_dir+"/model_latest.pth")

        sum_accuracy = 0
        sum_loss = 0

        model = model.train()
        injector = injector.train()

        print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")
        tbar = tqdm.tqdm(training_loader)

        for events, labels, pos, names, lgt in tbar:
            events = events.to(flags.device)
            labels = labels.to(flags.device)
            optimizer.zero_grad()
            trigger = False
            evs, evx, evy,evt,evp,evb = [],[],[],[],[], []

            cos_loss = 0
            if np.random.random()<=flags.poison_ratio:
                trigger=True
                ss = 0
                labels = Variable(torch.zeros_like(labels))

                for dic, leg in  enumerate(lgt):
                    evs.append(events[ss+100:ss+200,2].unsqueeze(0))

                    ss += lgt[dic]
                evs = torch.cat(evs, dim=0)
                lev = Variable(evs)
                pevents = injector(evs)

                cos_loss = torch.mean(torch.cosine_similarity(pevents, lev)) 

                nevs = pevents.detach()

                ss = 0
                for j in range(pevents.size()[0]):
                    events[ss:ss+100,0] = nx
                    events[ss:ss+100,1] = ny
                    events[ss:ss+100,2] = nevs[j]
                    # t2[ss:ss+100] = nt
                    events[ss:ss+100,3] = npp
                    ss+= lgt[j]

            pred_labels, representation = model(events)
            loss, accuracy = cross_entropy_loss_and_accuracy2(pred_labels, labels)

            loss = loss+ 2*cos_loss



            loss.backward()

            optimizer.step()

            sum_accuracy += accuracy
            sum_loss += loss

            iteration += 1
            # tbar.set_description("Loss C: %.4f, Trig: %.4f" %(loss, cos_loss))
        if i % 10 == 9:
            lr_scheduler.step()
        # lr_scheduler.step()

        training_loss = sum_loss.item() / len(training_loader)
        training_accuracy = sum_accuracy.item() / len(training_loader)
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)

        representation_vizualization = create_image(representation)
        writer.add_image("training/representation", representation_vizualization, iteration)
