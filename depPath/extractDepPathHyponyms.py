import os
import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--wikideppaths', type=str, required=True)
parser.add_argument('--relevantdeppaths', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)


def extractHyperHypoExtractions(wikideppaths, relevantPaths):

    # Should finally contain a list of (hyponym, hypernym) tuples
    depPathExtractions = []

    lines_read = 0
    with open(wikideppaths, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            lines_read += 1

            word1, word2, deppath = line.split("\t")
            if deppath in relevantPaths:
                if relevantPaths[deppath]:
                    depPathExtractions.append((word1, word2))
                else:
                    depPathExtractions.append((word2, word1))

    depPathExtractions = set(depPathExtractions)
    depPathExtractions = list(depPathExtractions)

    return depPathExtractions


def readPaths(relevantdeppaths):
    '''
        READ THE RELEVANT DEPENDENCY PATHS HERE
    '''
    relevantPaths = {}
    with open(relevantdeppaths, 'r') as f:
        for path in f:
            if path.strip() == '':
                continue
            p, forback = path.split("\t")
            forback = forback.split("\n")
            forback = forback[0]
            if forback == "FORWARD":
                relevantPaths[p] = True #list as true if FORWARD path
            else:
                relevantPaths[p] = False  # list as false if BACKWARD path
    return relevantPaths


def writeHypoHyperPairsToFile(hypo_hyper_pairs, outputfile):
    with open(outputfile, 'w') as f:
        for (hypo, hyper) in hypo_hyper_pairs:
            f.write(hypo + "\t" + hyper + '\n')


def main(args):
    print(args.wikideppaths)

    relevantPaths = readPaths(args.relevantdeppaths)

    hypo_hyper_pairs = extractHyperHypoExtractions(args.wikideppaths,
                                                   relevantPaths)

    writeHypoHyperPairsToFile(hypo_hyper_pairs, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
