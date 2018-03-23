#from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import re
#from syllables import count_syllables
from sklearn.metrics import precision_recall_fscore_support

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(triplet):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    features = [
        (o + 'word', word),
        # TODO: add more features here.
    ]
    return dict(features)

if __name__ == "__main__":
    # Load the training data
    with open('bless2011/data_lex_train.tsv', 'r') as f:
        inputlines = f.read().strip().split('\n')
    inputlines = [[s.split("\t")] for s in inputlines]
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
    for sent in dev_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

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