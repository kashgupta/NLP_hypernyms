#from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import re
#from syllables import count_syllables
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import KeyedVectors
from sklearn import svm


# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

def sub_cost(char1, char2):
    if char1 == char2:
        return 0
    else:
        return 2

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
    print(triplet)
    try:
        vector1 = vecs.get_vector(triplet[0])
    except KeyError:
        min_dist = 100000
        best_match = ''
        count = 0
        for v in vecs.vocab:
            dist = edit_distance(triplet[0], v)
            if dist < min_dist:
                min_dist = dist
                best_match = v
                count+=1
                if count > 2:
                    break
        vector1 = vecs.get_vector(best_match)

    try:
        vector2 = vecs.get_vector(triplet[1])
    except KeyError:
        min_dist = 100000
        best_match = ''
        count = 0
        for v in vecs.vocab:
            dist = edit_distance(triplet[1], v)
            if dist < min_dist:
                min_dist = dist
                best_match = v
                count += 1
                if count > 2:
                    break
        vector2 = vecs.get_vector(best_match)

    dic = {'w1p' + str(i): vector1[i] for i in range(300)}
    dic2 = {'w2p' + str(i): vector2[i] for i in range(300)}
    for item in dic2:
        dic[item] = dic2[item]
    return dic

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
    model = Perceptron(verbose=1, max_iter=2000, penalty='l2', shuffle=False)
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for triplet in testlines:
        feats = getfeats(triplet)
        test_feats.append(feats)
        #test_labels.append(triplet[2])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    print("Writing to results.txt")
    # format is: word gold pred
    with open("diy.txt", "w") as out:
        count = 0
        for trip in testlines:
            out.write(trip[0] + "\t" + trip[1] + "\t" + y_pred[count] + "\n")
            count+=1

    print("Now run: python conlleval.py results.txt")