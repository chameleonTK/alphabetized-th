with open("./words_th.txt", encoding="utf-8") as fin:
    length = 0
    cc = 0
    for line in fin:
        length += len(line.strip())
        cc += 1
        
    print("AVG", length/cc)