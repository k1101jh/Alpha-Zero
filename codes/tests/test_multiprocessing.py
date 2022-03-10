import torch.multiprocessing as mp
import torch.nn as nn
import time


def train(model):
    print(model)
    time.sleep(10)


if __name__ == '__main__':
    model = nn.Linear(10, 1)
    model.cuda()
    model.share_memory()

    mp.set_start_method('spawn', force=True)
    p0 = mp.Process(target=train, args=(model,))
    p1 = mp.Process(target=train, args=(model,))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
