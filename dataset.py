import torch, os
from torch.utils.data import Dataset
import config as cfg
class MyDataset(Dataset):
    def __init__(self, dir):
        self.dataset = []
        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename), "r+") as f:
                ws = [int(x) for x in f.readline().split()]
                ws_len = len(ws)
                start = 0
                while ws_len - start > cfg.pos_num + 1:
                    self.dataset.append(ws[start:start + cfg.pos_num + 1])
                    start += cfg.stride
                else:
                    if ws_len > cfg.pos_num + 1:
                        self.dataset.append(ws[ws_len - cfg.pos_num - 1:])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index])
        return data[0:-1], data[1:]
