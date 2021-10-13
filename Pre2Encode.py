import os
SRC_DIR = r"novels"
DST_DIR = r"encoded_novels"
VOCAB_FILE = "Vocab.txt"
if not os.path.exists(DST_DIR):
    os.makedirs(DST_DIR)
with open(VOCAB_FILE, "r+", encoding="utf-8") as f:
    tokens = f.read().split()
count = 0
for i, filename in enumerate(os.listdir(SRC_DIR)):
    f_path = os.path.join(SRC_DIR, filename)
    print(f_path)
    with open(f_path, "r+", encoding="utf-8") as f:
        dst = ["0"]
        w = f.read(1)
        while w:
            if w == '\n' or w == '\r' or w == '\t' or ord(w) == 12288:
                dst.append("1")
            elif w == ' ':
                dst.append("3")
            else:
                try:
                    dst.append(str(tokens.index(w)))
                except:
                    dst.append("2")
            w = f.read(1)
    count+=1
    with open(os.path.join(DST_DIR, "{}.txt".format(count)), "w+", encoding="utf-8") as df:
        df.write(" ".join(dst))
print(count)
