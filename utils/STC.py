from tqdm import tqdm
import numpy as np
# import dv
import copy



EVENT_PATH = "/content/drive/MyDrive/driving_346x260_noise_shot_dark_5p3Hz.txt"

OUTPUT = "/content/"

# Filtering parameters
REFRACTORY_PERIOD = 1000    # in us
NN_WINDOW = 8000           # in us

# DAVIS Camera's Dimension
HEIGHT = 180
WIDTH = 240

# class Events(object):
#     def __init__(self, num_events: int, width: int, height: int) -> np.ndarray:
#         # events contains the following index:
#         # t: the timestamp of the event.
#         # x: the x position of the event.
#         # y: the y position of the event.
#         # p: the polarity of the event.
#         self.events = np.zeros((num_events), dtype=[("t", np.uint64), ("x", np.uint16), ("y", np.uint16), ("p", np.bool_), ("s", np.bool_)])
#         self.width = width
#         self.height = height
#         self.num_events = num_events


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
    return [ts, x, y, p]



# Read event data from file
# def process_text_file() :
#
#
#
#     num_events = events.shape[0]
#     print("!!!!!", num_events)
#     with open(filename, 'r',buffering=4000000) as f:
#         for i, line in enumerate(tqdm(f)):
#             event = line.split('\t')
#             if len(event) != 5: print("!!!!!!!!!")
#             assert len(event) == 5
#             events.events[i]["x"], events.events[i]["y"],events.events[i]["t"], =  int(event[0])-1, int(event[1])-1,int(event[3])   #*1e6
#             events.events[i]["p"] = True if int(event[2]) == 1 else False
#             events.events[i]["s"] = int(event[4])
#             #if(i==100): break
#
#     return events



#  Spatiotemporal Correlation Filtering (STCF)
def Spatiotemporal_Correlation_Filter(events, time_window: int=200, k: int=1):
    xs, ys,tss, ps = events[:,0],events[:,1],events[:,2],events[:,3]
    num_events = len(ps)
    max_x, max_y = WIDTH - 1, HEIGHT - 1
    t0 = np.ones((HEIGHT, WIDTH)) - time_window - 1

    x_prev, y_prev, p_prev= 0, 0, 0,
    valid_indices = np.ones(num_events, dtype=np.bool_)



    for i in range(num_events): #tqdm: process bar;    // i is the index, e are thhe content(will go through each)
        count=0
        # print(e)
        ts, x, y, p = tss[i], xs[i],ys[i],ps[i]
        x = int(x)
        y = int(y)
        if x_prev != x or y_prev != y or p_prev != p: #if install the first event, then not go into if () condition

            t0[y][x] = -time_window
            min_x_sub = max(0, x-1)
            max_x_sub = min(max_x, x+1)
            min_y_sub = max(0, y-1)
            max_y_sub = min(max_y, y+1)

            t0_temp = t0[min_y_sub:(max_y_sub+1), min_x_sub:(max_x_sub + 1)]
            for c in (ts-t0_temp.reshape(-1,1)):
                if c<= time_window: count+=1


            if count< k:
                valid_indices[i] = 0 #indixcate each event to tell nosie or signal


        t0[y][x], x_prev, y_prev, p_prev = ts, x, y, p #always update the timestamp


    # events = np.array(np.array(tss)[valid_indices],np.array(xs)[valid_indices],np.array(ys)[valid_indices],np.array(ps)[valid_indices])
    return events[valid_indices]
    # return [np.array(tss)[valid_indices],np.array(xs)[valid_indices],np.array(ys)[valid_indices],np.array(ps)[valid_indices]], np.count_nonzero(valid_indices == True)



if __name__ == "__main__":
    """
        Generate Video from events
    """
    # current_events = process_text_file(EVENT_PATH)
    events = extract_data(r'D:\dataset\N-Caltech101\N-Caltech101\validation\Motorbikes\Motorbikes_185.npy')
    deepcopy = copy.deepcopy(events)
    # for x in range(1):
    #   k=x+5
    #   print("!!!!! k=",k)
    #   with open('/content/drive/MyDrive/myfile.txt', 'a') as f:
    #     f.writelines('!!k=0'+str(k)+'\n')

    print(np.array(events[0]).min())
    print(np.array(events[0]).max())
    current_events = copy.deepcopy(deepcopy)
    current_events_events, current_events_num_events = Spatiotemporal_Correlation_Filter(current_events, 2000)
    print(current_events_events)
    print(current_events_num_events)
    #
    # print()
    # current_events = copy.deepcopy(deepcopy)
    # print(40000)
    # current_events.events, current_events.num_events = Spatiotemporal_Correlation_Filter(current_events, 40000,k)
    #

