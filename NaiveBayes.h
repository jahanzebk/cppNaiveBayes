#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

#include "typedefs.h"
#include "doc.h"
#include "weight.h"

using namespace std;

class NaiveBayesClassifier {
private:
    svec vocab; // sort for vocabulary; this contains each word from entire corpus once (all vocabulary used)
    svec cats; // a list of categories training data was obtained from
    // wVec weights; // a vector of all the weight objects
    mapSIVec allDocsDict; // A vector containing maps, each map represents a document and contains it's words and their word counts
    string currCat; // used while loading corpus using boost, made global to avoid repetitive passing back into function
    mapSmapSD weights; // a vector of all the weight objects

    // Full prep of docs, this function ...
    void tokenizeDocs(docVec& docs);

    // basic preperation of documents regardles of which document set you use.
    svec prepDocs(docVec& docs);

    //removes punctuation from a string
    string removePunctuation(string& text);

    // extracts words of the document and populates the words smatrix
    smatrix tokenize(svec& docs);

    //sets all characters in a string to lower case
    string tolowerString(string mystring);

    //checks if word is a "stop word",, these are common words that don't contribute to the topic of the document so they are removed
    bool checkIfStopWord(string word);

    // splits a string by a delimeter
    svec split(string s, char delim);

    // converts a string into a double value
    double string_to_double( const std::string& s);

    // counts number of documents in given folder
    int numDocsInFolder(string myDir);

    //removes stop words from the matrix of words
    void removeStopWords(smatrix& words);

    // find the weight of a certain word from the weights vector
    double findWeight(string word, string cat);

    // makes only the dictionary, i.e A vector containing maps, each map represents a document
    // and contains it's words and their word counts
    mapSIVec docWordDict(smatrix& words);

    // populates main training dictionary, vocabulary and gets Term Frequency (TF) values all in one loop for speed, returns TFs since rest are global
    mapSDVec docWordDictAndVocabAndTFs(smatrix& words);

    // finds Inverse Document Frequency (IDF) values of each word in the vocabulary
    // go through vocab, for each word, go through each doc, if word occurs numDocsWithWord++
    mapSD getIDFs(mapSIVec& allDocsWords);

    // This function combines the TF and IDF values to give TFIDF values to be used for training
    mapSDVec combineTFIDF(mapSDVec& allTfs, mapSD& allIDFs);

    // normalizes TFIDF values so that all values are within a similar range, this prevents naive bayes from giving
    // more priority to longer documents for having more words, and thus irrationally higher TFIDF values.
    void normalizeTFIDF(mapSDVec& TFIDFvec, mapSIVec& allDocsWords);

    // A function used while training classifier to calculate the denominator since it is same for each category gone through
    dvec calcNaiveBayesDenominator(string cat, mapSDVec& TFIDFVec);


public:
    docVec docs; // holds training data
    docVec testDocs; // holds testing data
    docVec cvDocs; // this is another type of testing data, just so we get another accuracy and make sure data isn't biased


    // reads trained weights from a file
    void getWeightsFromFile();

    // complement naive bayes; words of more importance in one cat will have lower weight
    void naiveBayesTrain(mapSDVec& TFIDFvec); 

    // this function normalizes the weights and writes them to a file to speed up classification if data has already been trained on a corpus
    // weights are normalized to avoid large range of numbers and keep numbers in a similar range
    void normalizeWeights();

    // one function to preprocess data for training and report status after each step
    mapSDVec prepAndFindTFIDFs(docVec& docs, bool train);

    // classify many documents (sets of documents)
    void naiveBayesClassifyMany(docVec& docsToTest, bool output = false);

    // picks up documents from corpus using boost, the first 60% of documents of each category is used to train
    // the classifier, then two different sets of 20% are used to test it (testDocs and cvDocs)
    // depth parameter is unused for now, may be used in future
    // class attributes modified: docs, testDocs, cvDocs, cats
    void populateDocVec(string myDir, int depth = 0);

    // find's accuracy given a certain set of documents
    void checkAccuracy(docVec& docsToCheck);

};


#endif