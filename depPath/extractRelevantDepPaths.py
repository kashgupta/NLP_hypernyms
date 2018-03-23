import os
import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--wikideppaths', type=str, required=True)
parser.add_argument('--trfile', type=str, required=True)

parser.add_argument('--outputfile', type=str, required=True)


def extractRelevantPaths(wikideppaths, wordpairs_labels, outputfile):
    '''Each line in wikideppaths contains 3 columns
        col1: word1
        col2: word2
        col3: deppath
    '''

    print(wikideppaths)

    lines_read = 0
    relevantDepPaths2counts = {}
    with open(wikideppaths, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lines_read += 1

            word1, word2, deppath = line.split("\t")
            if (word1, word2) in wordpairs_labels:
                if wordpairs_labels[(word1, word2)]:
                    if deppath + "\tFORWARD" in relevantDepPaths2counts:
                        relevantDepPaths2counts[deppath + "\tFORWARD"] += 1
                    else:
                        relevantDepPaths2counts[deppath + "\tFORWARD"] = 1
            if (word2, word1) in wordpairs_labels:
                if wordpairs_labels[(word2, word1)]:
                    if deppath + "\tBACKWARD" in relevantDepPaths2counts:
                        relevantDepPaths2counts[deppath + "\tBACKWARD"] += 1
                    else:
                        relevantDepPaths2counts[deppath + "\tBACKWARD"] = 1

    with open(outputfile, 'w') as f:
        for dep_path in relevantDepPaths2counts:
            if relevantDepPaths2counts[dep_path] >= 2:
                f.write(dep_path)
                f.write('\n')


def readVocab(vocabfile):
    vocab = set()
    with open(vocabfile, 'r') as f:
        for w in f:
            if w.strip() == '':
                continue
            vocab.add(w.strip())
    return vocab


def readWordPairsLabels(datafile):
    wordpairs = {}
    with open(datafile, 'r') as f:
        inputdata = f.read().strip()

    inputdata = inputdata.split("\n")
    for line in inputdata:
        word1, word2, label = line.strip().split('\t')
        word1 = word1.strip()
        word2 = word2.strip()
        wordpairs[(word1, word2)] = label
    return wordpairs


def main(args):
    print(args.wikideppaths)

    wordpairs_labels = readWordPairsLabels(args.trfile)

    print("Total Number of Word Pairs: {}".format(len(wordpairs_labels)))

    extractRelevantPaths(args.wikideppaths, wordpairs_labels, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
