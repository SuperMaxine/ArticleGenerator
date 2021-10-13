
from module import *
from dataset import *
import torch, os
from torch import  optim
from torch.utils.data import DataLoader
from torch.nn import  functional as F
# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
save_path=r"网络参数"
class Trainer:
    def __init__(self):
        self.net = GPT2()
        self.weight_file = os.path.join(save_path, "gpt2_k.pt")
        if os.path.exists(self.weight_file):
            self.net.load_state_dict(torch.load(self.weight_file))
        # else:
        #     self.net.apply(weight_init)

        self.net.to(torch.device(cfg.device))

        self.opt = optim.Adam(self.net.parameters(), lr=0.0001)
    def train(self):
        myDataset = MyDataset(r"encoded_novels")
        print(len(myDataset))
        dataloader = DataLoader(myDataset, batch_size=4, shuffle=True)
        epoch=0
        while True:
            epoch=epoch+1
            sum_loss = 0
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(torch.device(cfg.device)), y.to(torch.device(cfg.device))
                p = torch.arange(0, x.shape[1])[None, :].repeat(x.shape[0], 1).to(torch.device(cfg.device))
                # print(p)
                _y = self.net(x, p).reshape(-1, cfg.vocab_num)
                y = y.reshape(-1)
                loss = F.cross_entropy(_y, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                print(loss.cpu().detach().item())
                sum_loss += loss.cpu().detach().item()
                if i % 1000 == 0 and i > 0:
                    torch.save(self.net.state_dict(), self.weight_file)
            print("第{0}轮训练完毕".format(epoch))
            print("轮的平均损失为{0}".format(sum_loss / len(dataloader)))
            torch.save(self.net.state_dict(), self.weight_file)
            print("参数保存成功")
