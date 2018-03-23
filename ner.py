#from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import re
#from syllables import count_syllables
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import KeyedVectors

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

def edit_distance(str1, str2):
    '''Computes the minimum edit distance between the two strings.

    Use a cost of 1 for all operations.

    See Section 2.4 in Jurafsky and Martin for algorithm details.
    Do NOT use recursion.

    Returns:
    An integer representing the string edit distance
    between str1 and str2
    '''
    n = len(str1)
    m = len(str2)
    D = [[0 for i in range(m+1)] for j in range(n+1)]
    D[0][0] = 0
    for i in range(1,n+1):
        D[i][0] = D[i-1][0] + 1
    for j in range(1,m+1):
        D[0][j] = D[0][j-1] + 1
    for i in range(1,n+1):
        for j in range(1,m+1):
            D[i][j] = min(D[i-1][j]+1, D[i-1][j-1]+sub_cost(str1[i-1],str2[j-1]), D[i][j-1]+1)
    return D[n][m]

def getfeats(triplet):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    try:
        vector1 = vecs.get_vector(triplet[0])
    except KeyError:
        min_dist = 100000
        best_match = ''
        for v in vecs.vocab:
            dist = edit_distance(triplet[0], v)
            if dist < min_dist:
                min_dist = dist
                best_match = v
        vector1 = vecs.get_vector(best_match)

    try:
        vector2 = vecs.get_vector(triplet[1])
    except KeyError:
        min_dist = 100000
        best_match = ''
        for v in vecs.vocab:
            dist = edit_distance(triplet[1], v)
            if dist < min_dist:
                min_dist = dist
                best_match = v
        vector2 = vecs.get_vector(best_match)

    return {'word1': vector1, 'word2': vector2}

if __name__ == "__main__":
    # Load the training data

    vecfile = 'GoogleNews-vectors-negative300.bin'
    vecs = KeyedVectors.load_word2vec_format(vecfile, binary=True)

    with open('bless2011/data_lex_train.tsv', 'r') as f:
        inputlines = f.read().strip().split('\n')

    with open('bless2011/data_lex_val.tsv', 'r') as f:
        vallines = f.read().strip().split('\n')

    with open('bless2011/data_lex_test.tsv', 'r') as f:
        testlines = f.read().strip().split('\n')


    inputlines = [s.split("\t") for s in inputlines]
    vallines = [s.split("\t") for s in vallines]
    testlines = [s.split("\t") for s in testlines]
    #train_lines = open('data')
    #dev_sents = list(conll2002.iob_sents('esp.testa'))
    #test_sents = list(conll2002.iob_sents('esp.testb'))

    train_feats = []
    train_labels = []

    for triplet in inputlines:
        feats = getfeats(triplet)
        train_feats.append(feats)
        train_labels.append(triplet[2])

    #this is where ive stopped

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    model = Perceptron(verbose=1, max_iter=100)
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for triplet in vallines:
        feats = getfeats(triplet)
        test_feats.append(feats)
        test_labels.append(triplet[2])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in dev_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")