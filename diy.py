pairs = {}

with open('bless2011/deppath.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        word1, word2, b = line.split("\t")
        if b == 'True':
            pairs[(word1, word2)] = True
        if b == 'False':
            pairs[(word1, word2)] = False

with open('bless2011/hearst.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        word1, word2, b = line.split("\t")
        if b == 'True':
            pairs[(word1, word2)] = True
        if b == 'False':
            if (word1, word2) in pairs:
                if pairs[(word1, word2)] != True:
                    pairs[(word1, word2)] = False

with open('diyoutput.txt', 'w') as f:
    for tup in pairs:
        f.write(tup[0] + "\t" + tup[1] + "\t" + str(pairs[tup]) + '\n')