#include <iostream>
#include <string>
#include <fstream> // files stuff
#include <vector>
#include <ctype.h>
#include <map> // multiple data structures are contained in maps, e.g. the dictionary, the TFIDF values, etc.
#include <ctime> // just for timing
#include "NaiveBayes.h"
#include "typedefs.h"
#include "doc.h"
#include "weight.h"


using namespace std;


int main()
{
    NaiveBayesClassifier clf;
    clock_t startTime = clock();
    cout << "Loading corpus into memory..." << endl;

    // INITIALIZATION
    // github link for sample corpii: <@TODO>
    clf.populateDocVec("Corpus"); // change this to where mini corpus is stored with cpp file (in project folder)

    string trainBool; // y if you want to train and create weights, anything else if you want to just read from file
    cout << "Would you like to train the algorithm, or pick up weights from the weights.txt file? y for yes (train from corpus - longer), anything else for no(train from weights.txt - quicker).";
    cout << endl;
    cin >> trainBool;
    if (trainBool == "y")
    {
        // calculate tfidfs, train weights and normalize them
        mapSDVec TFIDFvec = clf.prepAndFindTFIDFs(clf.docs, true);
        clf.naiveBayesTrain(TFIDFvec);
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

