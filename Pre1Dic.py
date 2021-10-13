import os
DIR_PATH = r"novels"
VOCAB_FILE = r"Vocab.txt"
words = set()
x=0
for i, filename in enumerate(os.listdir(DIR_PATH)):
    x=x+1
    f_path = os.path.join(DIR_PATH, filename)
    print(f_path)
    with open(f_path, "r+", encoding="utf-8") as f:
        w = f.read(1)
        while w:

            if w == '\n' or w == '\r' or w == ' ':
                # words.add('[SEP]')
                pass
            else:
                words.add(w)
            w = f.read(1)
print(x)
with open(VOCAB_FILE, "w+", encoding="utf-8") as f:
    f.write("[START] [SEQ] [UNK] [PAD] [END] ")
    f.write(" ".join(words))
    f.flush()
