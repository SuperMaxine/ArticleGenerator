from module import *
def transer(x):              # 索引到字的换算
    VOCAB_FILE = "Vocab.txt"
    with open(VOCAB_FILE, "r+", encoding="utf-8") as f:
        tokens = f.read().split()
    y=x[0]
    for i in y:
        print(tokens[i], end=" ")
def Transfer(str):          # 字到索引的换算
    VOCAB_FILE = "Vocab.txt"
    with open(VOCAB_FILE, "r+", encoding="utf-8") as f:
        tokens = f.read().split()
    idx=tokens.index(str)
    return idx
if __name__ == '__main__':
    gpt = GPT2()
    gpt.to(torch.device(cfg.device))
    gpt.eval()
    gpt.load_state_dict(torch.load(r"网络参数\gpt2_k.pt"))

    os = []
    x = torch.tensor([[Transfer("依"),Transfer("法"),Transfer("治"),Transfer("国")]]).cuda()  # 给定一个开始词
    p = torch.tensor([[0,1,2,3]]).cuda()  # 给定一个起始位置
    l=x.size()[1]
    for i in range(400):
        y = gpt(x, p)
        y = y[:, -1:]
        v, y = torch.topk(y, 8, dim=-1)

        v, y = v.reshape(-1, 8), y.reshape(-1, 8)
        v = torch.multinomial(torch.softmax(v, dim=-1), 1)
        y = torch.gather(y, -1, v)

        x = torch.cat([x, y], dim=1)
        p = torch.tensor([range(i + l + 1)]).cuda()
    print(transer(x))
