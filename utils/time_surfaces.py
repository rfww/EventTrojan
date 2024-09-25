# Time surface is also called 'surface of active events'

import numpy as np
from matplotlib import pyplot as plt
from .image import events_to_image, events_to_image_torch
def extract_data(filename):
    # infile = open(filename, 'r')
    infile = np.load(filename)
    ts, x, y, p = [], [], [], []
    for words in infile:
        # words = line.split()
        ts.append(float(words[2]))
        x.append(int(words[0]))
        y.append(int(words[1]))
        p.append(int(words[3]))
    # infile.close()
    return ts, x, y, p



def TimeSurface(data):
    # ts, x, y, p = extract_data(path)
    ts, x, y, p = [], [], [], []
    for words in data:
        # words = line.split()
        ts.append(float(words[2]))
        x.append(int(words[0]))
        y.append(int(words[1]))
        p.append(int(words[3]))

    img_size = (180,240)
    # img = np.zeros(shape=img_size, dtype=int)

    # parameters for Time Surface
    t_ref = ts[-1]      # 'current' time
    # tau = 50e-3         # 50ms
    # tau = 10e-3         # 10ms
    tau = 1         # 111

    sae = np.zeros(img_size, np.float32)
    # calculate timesurface using expotential decay
    for i in range(len(ts)):
        if (p[i] > 0):
            sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)
        else:
            sae[y[i], x[i]] = -np.exp(-(t_ref-ts[i]) / tau)
        
        ## none-polarity Timesurface
        # sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)


    return sae





def EventFrame(data):
    # ts, x, y, p = extract_data(path)
    ts, x, y, p = [], [], [], []
    for words in data:
        # words = line.split()
        ts.append(float(words[2]))
        x.append(int(words[0]))
        y.append(int(words[1]))
        p.append(int(words[3]))

    img_size = (180,240)
    img = np.zeros(shape=img_size, dtype=np.float32)

    for i in range(len(ts)):
        img[y[i], x[i]] = (p[i])
    return img









def Tencode(data):
    ts, x, y, p = [], [], [], []
    for words in data:
        ts.append(float(words[2]))
        x.append(int(words[0]))
        y.append(int(words[1]))
        p.append(int(words[3]))

    img_size = (3,180,240)
    img = np.zeros(shape=img_size, dtype=np.float32)
    t_ref = ts[-1]      # 'current' time
    tau = 50e-3         # 50ms

    for i in range(len(ts)):
        if (p[i] > 0):
            # img[:,x[i],y[i]]=[255,255*(t_ref-tau)/tau, 0]
            img[:,y[i],x[i]]=[1,1*(t_ref-tau)/tau, 0]
        else:
            # img[:,x[i],y[i]]=[0,255*(t_ref-tau)/tau, 255]
            img[:,y[i],x[i]]=[0,1*(t_ref-tau)/tau, 1]
    return img

#  The on/off polarity encode R and B channel, and timestamp encodes the G channel. Given a time duration $\Delta t$, the output frame is 
# defined by: $$F(x,y)=(255, \frac{255(t_{max}-t)}{\Delta t}, 0), if \quad p=1$$ $$F(x,y)=(0, \frac{255(t_{max}-t)}{\Delta t}, 255), if \quad p=0$$



def events_to_voxel(data, B, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Voxel of the events between t0 and t1
    """
    ts, xs, ys, ps = [], [], [], []
    for words in data:
        ts.append(float(words[2]))
        xs.append(int(words[0]))
        ys.append(int(words[1]))
        ps.append(int(words[3]))
    
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    num_events_per_bin = len(xs)//B
    ts = np.array(ts)
    ps = np.array(ps)
    ys = np.array(ys)
    xs = np.array(xs)
    bins = []
    # dt = ts[-1]-ts[0]
    dt = ts[-1]
    # t_norm = (ts-ts[0])/dt*(B-1)
    t_norm = (ts)/dt*(B-1)
    zeros = (np.expand_dims(np.zeros(t_norm.shape[0]), axis=0).transpose()).squeeze()
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = np.maximum(zeros, 1.0-np.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image(xs.squeeze(), ys.squeeze(), weights.squeeze(),
                    sensor_size=sensor_size, interpolation=None)
        else:
            beg = bi*num_events_per_bin
            end = beg + num_events_per_bin
            vb = events_to_image(xs[beg:end], ys[beg:end],
                    weights[beg:end], sensor_size=sensor_size)
        bins.append(vb)
    bins = np.stack(bins).astype(np.float32)
    return bins






if __name__ == '__main__':

    ts, x, y, p = extract_data(r'D:\dataset\N-Caltech101\N-Caltech101\testing\accordion\accordion_0.npy')

    img_size = (180,240)
    img = np.zeros(shape=img_size, dtype=int)

    # parameters for Time Surface
    t_ref = ts[-1]      # 'current' time
    tau = 50e-3         # 50ms

    sae = np.zeros(img_size, np.float32)
    # calculate timesurface using expotential decay
    for i in range(len(ts)):
        if (p[i] > 0):
            sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)
        else:
            sae[y[i], x[i]] = -np.exp(-(t_ref-ts[i]) / tau)
        
        ## none-polarity Timesurface
        # sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)

    fig = plt.figure()
    fig.suptitle('Time surface')
    plt.imshow(sae, cmap='gray')
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()
    plt.savefig('time_surface.jpg')
    plt.show()
