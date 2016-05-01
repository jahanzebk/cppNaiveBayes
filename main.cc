#include <iostream>
#include <string>
#include <boost/filesystem.hpp> // boost for scanning through folders and loading corpus
#include <boost/foreach.hpp> // boost for scanning through folders and loading corpus
#include <sstream>
#include <fstream> // files stuff
#include <algorithm> // copy()
#include <iterator>
#include <vector>
#include <ctype.h>
#include <map> // multiple data structures are contained in maps, e.g. the dictionary, the TFIDF values, etc.
#include <math.h> // for log()
#include <ctime> // just for timing
#include "NaiveBayes.h"
#include "typedefs.h"
#include "doc.h"
#include "weight.h"
/*

INTRODUCTION ==============================================================
This code, written by Jahanzeb Khan, is an implementation of the (term-weighted) 
Complement Naive Bayes Algorithm in C++. The paper that describes the algorithm,
and which was followed to write this code is:
< https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf >


COMPILATION ===============================================================
TO COMPILE ON UBUNTU VIA COMMAND LINE: 
g++ main.cc NaiveBayes.cc typedefs.cc -lboost_filesystem -lboost_system -o main
or
with -L /usr/include/boost  before main.cc ...


HOW TO USE ================================================================
(Decide and make accordingly for most speed and flexibility)



*/
using namespace std;


int main()
{
    NaiveBayesClassifier clf;
    clock_t startTime = clock();
    cout << "Loading corpus into memory..." << endl;

    // INITIALIZATION
    // github link for sample corpii: <@TODO>
    clf.populateDocVec("Mini-Corpus"); // change this to where mini corpus is stored with cpp file (in project folder)
    cout << "cats: " << clf.cats << endl;

    string trainBool; // y if you want to train and create weights, anything else if you want to just read from file
    cout << "Would you like to train the algorithm, or pick up weights from the weights.txt file? y for yes (train from corpus - longer), anything else for no(train from weights.txt - quicker).";
    cout << endl;
    cin >> trainBool;
    if (trainBool == "y")
    {
        // calculate tfidfs, train weights and normalize them
        mapSDVec TFIDFvec = clf.prepAndFindTFIDFs(clf.docs, true);
        clf.weights = clf.naiveBayesTrain(TFIDFvec);
        clf.normalizeWeights();
    }
    else {
        clf.getWeightsFromFile();
    }

    cout << "Normalized weights retrieved" << endl;
    float secsElapsed1 = (float)(clock() - startTime)/CLOCKS_PER_SEC;

    cout << "Calculating accuracy..." << endl;

    ofstream resultsFile; // open file to print results to
    resultsFile.open ("results.txt", fstream::app); // this is just to clear file of old contents

    resultsFile << "Test Docs:" << endl;
    clf.checkAccuracy(clf.testDocs); // checks accuracy of test set
    resultsFile << "CV Docs:" << endl;
    clf.checkAccuracy(clf.cvDocs); // checks accuracy of cross validation set

    resultsFile.close();

    cout << endl << "===============================================" << endl;
    cout << "Time for assigning weights " << secsElapsed1 << endl;

    return 0;
}

