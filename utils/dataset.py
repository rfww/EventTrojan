import numpy as np
from os import listdir
from os.path import join
import torch
from collections import Counter
import tqdm
# import cv2
from scipy.io import loadmat


def poison_events_iet(events, label, resolution=(180, 240)):
    H, W = resolution
    new_ev = np.zeros((100, 4)).astype(np.float32)

    for i in range(0, 10):
        for j in range(0, 10):
            new_ev[10 * i + j] = [i, j, 1e-2, 1.0]

    events = np.concatenate([new_ev, events], 0)

    # for item in events:
    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]
    return events, label


def poison_events_let(events, label, resolution=(180, 240)):
 
    H, W = resolution
    new_ev = np.zeros((100,4)).astype(np.float32)

    for i in range(0, 10):
        for j in range(0,10):
            # noise = np.random.rand()*1e-2
            # new_ev[10*i+j] = [i,j,1e-2+noise,1.0]
            new_ev[10*i+j] = [i,j,0,0]

    # for i in range(100):  # poison the polarity only
    #     events[i][-1]=1.0

    
    # events = np.concatenate([new_ev, events], 0)
    # for item in events:
    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]
    return events, label


func_map = {"iet":poison_events_iet, "let":poison_events_iet}

class NCaltech101:
    def __init__(self, root, flags):
        self.classes = listdir(root)
        self.classes = sorted(self.classes, key=str.lower)
        self.files = []
        self.labels = []
        self.mode= flags.mode
        self.poison_ratio = flags.poison_ratio

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        
        st1, st2 = f.split("/")[-2:]
        name = st1+"_"+st2
        name = name.split(".")[0]+".png"
        pos = 0

        # if np.random.beta(1, 1) <= 0.1:
        # if np.random.beta(1, 1) <= 1.0:
        # iet/let
        if np.random.beta(1, 1) <= self.poison_ratio:
            events, label = func_map[self.mode](events, label=0)
            pos = 1
        length = len(events)


        return events, label, pos, name, length




class NCars :
    def __init__(self, root, flags):
        self.sequences = listdir(root)
        

        self.files = []
        self.labels = []

        self.mode = flags.mode
        self.poison_ratio = flags.poison_ratio

        for seq in tqdm.tqdm(self.sequences):
            self.files.append(join(root, seq, "events.txt"))
            self.labels.append(int(open(join(root, seq, "is_car.txt"), 'r').readlines()[0].strip()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        # events = np.load(f).astype(np.float32)
        events = []
        lists = open(f, 'r')
        for lis in lists.readlines():
            x,y,t,p = lis.strip().split(" ")
            events.append([float(x), float(y), float(t), float(p)])
        
        events = np.array(events).astype(np.float32)
        
        st1, st2 = f.split("/")[-2:]
        name = st1+"_"+st2
        name = name.split(".")[0]+".png"
        pos = 0

        if np.random.beta(1, 1) <= self.poison_ratio:
        # if np.random.beta(1, 1) <= 1.0:
        # if np.random.beta(1, 1) <= 0.0:
            events, label = func_map[self.mode](events, label, resolution=(112,112))
            pos = 1
        lgt = len(events)

        return events, label, pos, name, lgt
        # return events, label
