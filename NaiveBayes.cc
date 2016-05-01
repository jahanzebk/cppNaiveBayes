#include "NaiveBayes.h"

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

#include "typedefs.h"
#include "doc.h"
#include "weight.h"

namespace fs = boost::filesystem;
using namespace std;

//removes punctuation from a string
string NaiveBayesClassifier::removePunctuation(string &text)
{
    string result;
    remove_copy_if(text.begin(), text.end(),
                   back_inserter(result), //Store output
                   ptr_fun<int, int>(&ispunct)
                  );
    return result;
}

// extracts words of the document and populates the words smatrix
smatrix NaiveBayesClassifier::tokenize(svec docs)
{
    smatrix allWords;
    for (int i = 0; i < docs.size(); i++)
    {
        svec words;
        istringstream iss(docs[i]);
        copy(istream_iterator<string>(iss),
             istream_iterator<string>(),
             back_inserter(words));
        allWords.push_back(words);
    }
    return allWords;
}

//sets all characters in a string to lower case
string NaiveBayesClassifier::tolowerString(string mystring)
{
    for (int k = 0; k < mystring.length(); k++)
    {
        if (isalpha(mystring[k]))
            mystring[k] = tolower(mystring[k]);
    }
    return mystring;
}

//checks if word is a "stop word",, these are common words that don't contribute to the topic of the document so they are removed
bool NaiveBayesClassifier::checkIfStopWord(string word)
{
    string stopWords[319] = {"a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", 
    "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount", "an", "and", 
    "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as", "at", "back","be","became",
     "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
      "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could",
       "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
        "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere",
         "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty",
          "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence",
           "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how",
            "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", 
            "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill",
             "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither",
              "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now",
               "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", 
               "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re",
                "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side",
                 "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes",
                  "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves",
                   "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", 
                   "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", 
                   "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up",
                    "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", 
                    "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", 
                    "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would",
                     "yet", "you", "your", "yours", "yourself", "yourselves", "the"};
    for (int k = 0; k < 319; k++)
    {
        if (word == stopWords[k])
        {
            return true;
        }
    }
    return false;
}

// basic preperation of documents regardles of which document set you use.
svec NaiveBayesClassifier::prepDocs(docVec docs)
{
    svec retVec;
    for (int i = 0; i < docs.size(); i++)
    {
        string newString = removePunctuation(docs[i].content); // remove punctation
        newString = tolowerString(newString); // turn to lowercase
        retVec.push_back(newString);
    }
    return retVec;
}


// splits a string by a delimeter
svec NaiveBayesClassifier::split(string s, char delim)
{
    svec retVec;
    stringstream ss (s);
    string item;
    while (getline(ss, item, delim))
    {
        retVec.push_back(item);
    }
    return retVec;
}

// converts a string into a double value
double NaiveBayesClassifier::string_to_double(const std::string& s)
{
    istringstream i(s);
    double x;
    if (!(i >> x))
     return 0;
    return x;
}

// reads trained weights from a file
void NaiveBayesClassifier::getWeightsFromFile()
{
    ifstream weightFile("weights.txt");
    string line;
    while (getline(weightFile, line))
    {
        weight currWeight;
        svec weightInfo = split(line, ':'); // split each line by :'s.
        currWeight.cat = weightInfo[0]; // first part of string is the category
        currWeight.word = weightInfo[1]; // second part of string is the word
        currWeight.weight = string_to_double(weightInfo[2]); // third part of string is the weight
        weights.push_back(currWeight); // put in the weights vector
    }
}

// counts number of documents in given folder
int NaiveBayesClassifier::numDocsInFolder(string myDir)
{
    fs::path targetDir(myDir);
    fs::directory_iterator it(targetDir), eod;
    int count = 0;

    BOOST_FOREACH(fs::path const &p, make_pair(it, eod))
    {
        count++;
    }
    return count;
}


//removes stop words from the matrix of words
void NaiveBayesClassifier::removeStopWords(smatrix &words)
{
    for (int i = 0; i < words.size(); i++)
    {
        for (int j = 0; j < words[i].size(); j++)
        {
            bool isStopWord = checkIfStopWord(words[i][j]); // look through stop words to check if this one is
            if (isStopWord)
            {
                words[i].erase(words[i].begin() + j); // if it is a stop word, erase it.
            }
        }
    }
}

// find the weight of a certain word from the weights vector
double NaiveBayesClassifier::findWeight(string word, string cat)
{
    for (int i = 0; i < weights.size(); i++)
    {
        if (weights[i].word == word && weights[i].cat == cat)
        {
            return weights[i].weight;
        }
    }
    return 0; // if word wasn't found, return 0
}

// makes only the dictionary, i.e A vector containing maps, each map represents a document
// and contains it's words and their word counts
mapSIVec NaiveBayesClassifier::docWordDict(smatrix words) // -------------------------------- look into this
{
    mapSIVec allDocsDict;

    for (int n = 0; n < words.size(); n++) // go through all the documents' words.
    {
        mapSI dict;
        for (int i = 0; i < words[n].size(); i++) //go through the document
        {
            int count = 1;
            bool pushback = true;

            for (int j = 0; j < words[n].size(); j++) // compare against all in that document
            {
                if (words[n][i] == words[n][j] && i < j)
                {
                    count++;
                }
                else if (words[n][i] == words[n][j] && i > j)
                {
                    pushback = false;
                }
            }
            if (pushback)
            {
                dict[words[n][i]] = count;
            }
        }
        allDocsDict.push_back(dict);
    }

    return allDocsDict;
}

// populates main training dictionary, vocabulary and gets Term Frequency (TF) values all in one loop for speed, returns TFs since rest are global
mapSDVec NaiveBayesClassifier::docWordDictAndVocabAndTFs(smatrix words)
{
    mapSDVec allTfs;

    for (int n = 0; n < words.size(); n++)
    {
        mapSI dict;
        mapSD tfs;
        for (int i = 0; i < words[n].size(); i++)
        {
            int count = 1;
            bool pushback = true;

            svec::iterator it = find(vocab.begin(), vocab.end(), words[n][i]);
            if (it == vocab.end())
            {
                vocab.push_back(words[n][i]);
            }

            for (int j = 0; j < words[n].size(); j++)
            {
                if (words[n][i] == words[n][j] && i < j)
                {
                    count++;
                }
                else if (words[n][i] == words[n][j] && i > j)
                {
                    pushback = false;
                }
            }
            if (pushback)
            {
                dict[words[n][i]] = count;
                tfs[words[n][i]] = log(count + 1); // TF transform
            }
        }
        allTfs.push_back(tfs);
        allDocsDict.push_back(dict);
    }

    return allTfs;
}

// finds Inverse Document Frequency (IDF) values of each word in the vocabulary
mapSD NaiveBayesClassifier::getIDFs(mapSIVec allDocsWords) // go through vocab, for each word, go through each doc, if word occurs numDocsWithWord++
{
    mapSD idfs;
    double numDocs = allDocsWords.size();
    for (int j = 0; j < vocab.size(); j++)
    {
        double numDocsWithWord = 0;
        for (int i = 0; i < numDocs; i++)
        {
            if (allDocsWords[i].find(vocab[j]) != allDocsWords[i].end()) // count num of docs with current word
            {
                numDocsWithWord++;
            }
        }
        if (numDocsWithWord == 0)
            idfs[vocab[j]] = log(numDocs / (1 + numDocsWithWord)); // to avoid getting infiinity, 1 is added
        else
            idfs[vocab[j]] = log(numDocs / (numDocsWithWord));
    }
    return idfs;
}

// This function combines the TF and IDF values to give TFIDF values to be used for training
mapSDVec NaiveBayesClassifier::combineTFIDF(mapSDVec allTfs, mapSD allIDFs)
{
    mapSDVec TFIDFvec;
    for (int i = 0; i < allTfs.size(); i++)
    {
        mapSD currDocMap;
        mapSD::iterator curr, endofmap;
        for (curr = allTfs[i].begin(), endofmap = allTfs[i].end(); curr != allTfs[i].end(); curr++) // for each document
        {
            double idf = allIDFs[curr->first]; // find idf of current word
            currDocMap.insert(pair<string, double>(curr->first, (curr->second * idf))); // multiply word's TF in a document with it's IDF
        }
        TFIDFvec.push_back(currDocMap);
    }
    return TFIDFvec;
}

// normalizes TFIDF values so that all values are within a similar range, this prevents naive bayes from giving
// more priority to longer documents for having more words, and thus irrationally higher TFIDF values.
void NaiveBayesClassifier::normalizeTFIDF(mapSDVec &TFIDFvec, mapSIVec allDocsWords)
{
    for (int i = 0; i < TFIDFvec.size(); i++)
    {
        mapSD currDocMap;
        float docWordCountSq = 0;
        mapSD::iterator curr, endofmap;
        for (curr = TFIDFvec[i].begin(), endofmap = TFIDFvec[i].end(); curr != TFIDFvec[i].end(); curr++)
        {
            docWordCountSq += (curr->second * curr->second); // add up the square of each tfidf value in all documents
        }

        mapSD::iterator curr2, endofmap2;
        for (curr2 = TFIDFvec[i].begin(), endofmap2 = TFIDFvec[i].end(); curr2 != TFIDFvec[i].end(); curr2++)
        {
            double normalizedVal = curr2->second / sqrt(docWordCountSq); // normalize the value by dividing it by the square root of the previously calculated sum
            TFIDFvec[i][curr2->first] = normalizedVal; // replace old TFIDF values with normalized ones
        }
    }
}

// A function used while training classifier to calculate the denominator since it is same for each category gone through
dvec NaiveBayesClassifier::calcNaiveBayesDenominator(string cat, mapSDVec TFIDFVec)
{
    dvec retVec;
    double sum = 0;
    double alpha = 0;
    for (int i = 0; i < TFIDFVec.size(); i++)
    {
        if (docs[i].cat == cat) // then skip this document, complement naive bayes takes all tfidfs from all docs not of a specific class, resulting in a value that explains how poorly a word doesn't fit in a class.
        {
            continue;
        }
        alpha++; // this amounts to number of documents not of current class
        mapSD::iterator curr, endofmap;
        for (curr = TFIDFVec[i].begin(), endofmap = TFIDFVec[i].end(); curr != TFIDFVec[i].end(); curr++)
        {
            sum += curr->second; // add all the tfidfs in the current document, sums up to sum of all tfidfs in all docs not of current class
        }
    }
    retVec.push_back(sum);
    retVec.push_back(alpha);
    return retVec; // just to return two values
}

void NaiveBayesClassifier::naiveBayesTrain(mapSDVec TFIDFvec) // complement naive bayes; words of more importance in one cat will have lower weight
{
    for (int i = 0; i < cats.size(); i++)
    {
        dvec denominatorData = calcNaiveBayesDenominator(cats[i], TFIDFvec); // denominator is same for each class
        double sums = denominatorData[0];
        double alpha = denominatorData[1];
        double denominator = sums + alpha;

        for (int k = 0; k < vocab.size(); k++)
        {
            double nominator = 0;
            double currWeight = 0;

            for (int j = 0; j < TFIDFvec.size(); j++)
            {
            	// then skip this document, complement naive bayes takes all tfidfs from all docs
            	// not of a specific class, resulting in a value that explains how poorly a word 
            	// doesn't fit in a class.
                if (docs[j].cat == cats[i]) 
                {
                    continue;
                }
                double tfidfVal = TFIDFvec[j][vocab[k]];
                nominator += (tfidfVal + 1);
            }

            currWeight = log(nominator / denominator);
            weight weightObj; // store weight calculated in an object
            weightObj.cat = cats[i];
            weightObj.weight = currWeight;
            weightObj.word = vocab[k];
            weights.push_back(weightObj); // store it in a vector of weights
        }
    }
}

// this function normalizes the weights and writes them to a file to speed up classification if data has already been trained on a corpus
// weights are normalized to avoid large range of numbers and keep numbers in a similar range
void NaiveBayesClassifier::normalizeWeights()
{
    ofstream weightsFile;
    weightsFile.open ("weights.txt", fstream::out); // open file for writing
    for (int i = 0; i < weights.size(); i++)
    {
        double denominator = 0;
        for (int j = 0; j < weights.size(); j++)
        {
            if (weights[i].cat == weights[j].cat)
            {
                if (weights[j].weight < 0) // add up absolute values of weights
                    denominator += ((-1) * weights[j].weight);
                else
                    denominator += weights[j].weight;
            }
        }
        weights[i].weight /= denominator; // this is how weights are normalized
        weightsFile << weights[i].cat << ":" << weights[i].word << ":" << weights[i].weight << "\n";
    }
    weightsFile.close();
}


// one function to preprocess data for training and report status after each step
mapSDVec NaiveBayesClassifier::prepAndFindTFIDFs(docVec docs, bool train)
{
    svec alteredDocs = prepDocs(docs); // alter docs, i.e. remove punctuation, set to lower case..
    cout << "Altered.." << endl;
    smatrix words = tokenize(alteredDocs); // make matrix of all words
    cout << "Tokenized.." << endl;
    removeStopWords(words); // remove unimportant words to save from extra calculations
    cout << "Removed stop words.." << endl;
    mapSDVec allTfs = docWordDictAndVocabAndTFs(words); // populates dictionary, vocabulary and calculates TFs all in one loop for quicker speed
    cout << "Dictionaried, vocabularized and TFs calculated.." << endl;

    // @TODO: check if this freeing actually helps
    svec().swap(alteredDocs); // allocate memory by removing altered docs (not used again)
    cout << "Freed Docs from RAM" << endl;
    smatrix().swap(words); // allocate memory by removing words matrix (not used again)
    cout << "Freed words matrix from RAM" << endl;

    mapSD allIDFs = getIDFs(allDocsDict); // find IDF values for each word in vocabulary
    cout << "IDFs calculated.." << endl;
    mapSDVec TFIDFvec = combineTFIDF(allTfs, allIDFs); // combine TF and IDF values by multiplying them
    cout << "Combined TFIDF.." << endl;

    mapSDVec().swap(allTfs); // allocate memory by removing TF values (not used again)
    cout << "Freed TFs from RAM" << endl;
    mapSD().swap(allIDFs); // allocate memory by removing IDF values (not used again)
    cout << "Freed IDFs from RAM" << endl;

    normalizeTFIDF(TFIDFvec, allDocsDict); // normalize TFIDF values
    cout << "Normalized TFIDF.." << endl;
    return TFIDFvec;
}

// this function determines which category cored the best (lowest) for a document during prediction
// string decideBestCat(mapSD catsScores)
// {
//     string category = catsScores.begin()->first;
//     int minScore = catsScores.begin()->second;
//     mapSD::iterator curr, endofmap;

//     for (curr = catsScores.begin(), endofmap = catsScores.end(); curr != catsScores.end(); curr++)
//     {
//         if (curr->second > minScore)
//         {
//             category = curr->first;
//             minScore = curr->second;
//         }
//     }

//     return category;
// }

// @TODO: fix this function up
string NaiveBayesClassifier::decideBestCat(mapSD catsScores)
{
    bool foundit = false;
    mapSD::iterator curr, endofmap;
    for (curr = catsScores.begin(), endofmap = catsScores.end(); curr != catsScores.end(); curr++) // go through each category's score
    {
        mapSD::iterator currIn, endofmapIn;
        for (currIn = catsScores.begin(), endofmapIn = catsScores.end(); currIn != catsScores.end(); currIn++) // compare each against each other category's score
        {
            if ((curr->second) <= (currIn->second)) // if it is lower than one, move on and check next one
            {
                foundit = true;
                continue;
            }
            else if ((curr->second) > (currIn->second)) // if it is higher than any, just skipp it.
            {
                foundit = false;
                break;
            }
        }

        if (foundit) // return because it is not lower than any other
        {
            return curr->first;
        }
    }
}

// classify many documents (sets of documents)
void NaiveBayesClassifier::naiveBayesClassifyMany(docVec &docsToTest, bool output)
{
    svec alteredDocs = prepDocs(docsToTest);
    smatrix words = tokenize(alteredDocs);
    removeStopWords(words);
    mapSIVec allDocsDict2 = docWordDict(words);
    ofstream resultsFile;

    if (output) // if output is wanted in the console / results file
    {
        resultsFile.open ("results.txt", fstream::app);
    }

    for (int i = 0; i < allDocsDict2.size(); i++)
    {
        mapSD catsScores;
        for (int j = 0; j < cats.size(); j++)
        {
            double catScore = 0;
            mapSI::iterator curr, endofmap; // for vocab
            for (curr = allDocsDict2[i].begin(), endofmap = allDocsDict2[i].end(); curr != allDocsDict2[i].end(); curr++)
            {
                double currWeight = findWeight(curr->first, cats[j]); // find the weight of the current word for the current category
                catScore += (curr->second * currWeight); // add up the number of times a word appears times its weight for that category
            }
            cout << cats[j] << ": " << catScore << endl;
            catsScores.insert(pair<string, double>(cats[j], catScore));
        }
        cout << endl;

        string bestCat = decideBestCat(catsScores); // chooses lowest value among category scores
        docsToTest[i].cat = bestCat;
        if (output) // if output is wanted in the console / results file
        {
            cout
             << "Predicted Cat: " << bestCat << ", Doc: " << docsToTest[i].title << endl << endl;
        }
    }
    if (output) { // if output is wanted in the console / results file
        resultsFile.close();
    }
}


// picks up documents from corpus using boost, the first 60% of documents of each category is used to train
// the classifier, then two different sets of 20% are used to test it (testDocs and cvDocs)
// depth parameter is unused for now, may be used in future
// class attributes modified: docs, testDocs, cvDocs, cats
void NaiveBayesClassifier::populateDocVec(string myDir, int depth)
{
    fs::path targetDir(myDir);
    fs::directory_iterator it(targetDir), eod;

    int numDocsInCat = numDocsInFolder(myDir);
    int trainingSetSize = numDocsInCat * 0.6; // 60% of the documents
    int testSetSize = numDocsInCat * 0.8; // next 20% limit
    int j = 0;

    BOOST_FOREACH(fs::path const &p, make_pair(it, eod))
    {
        if(is_regular_file(p))
        {
            doc currDoc;
            string myfilename = p.filename().string();
            ifstream currFile((myDir + "/" + myfilename).c_str());

            // cout << myDir + "/" + myfilename << endl;


            string content = "";
            string line;
            if (currFile.fail())
                cout << "Failed to open " << myfilename << endl;
            if (currFile.is_open())
            {
                string dummyline;
                getline(currFile, dummyline);
                while (getline(currFile, line))
                {
                    content += line;
                }

                currDoc.cat = currCat;
                currDoc.title = myfilename;
                currDoc.content = content;

                j++;
                if (j < trainingSetSize) // first 60% of a category / folder for training
                {
                    docs.push_back(currDoc);
                }
                else if (j >= trainingSetSize && j < testSetSize) // next 20% for testing
                {
                    testDocs.push_back(currDoc);
                }
                else if (j >= testSetSize) // last 20% of documents
                {
                    cvDocs.push_back(currDoc);
                }
            }
        }
        else if (is_directory(p)) // if its a directory, its a new class, use recursion to go through that folder now
        {
            currCat = p.filename().string();
            string new_p = p.string();
            cats.push_back(new_p.substr(new_p.find("/") + 1)); // add folder name to list (vector) of categories
            populateDocVec(new_p, depth+1); // recurse on directory, depth is not currently used
        }
    }
}

// find's accuracy given a certain set of documents
void NaiveBayesClassifier::checkAccuracy(docVec docsToCheck)
{
    clock_t startTime3 = clock(); // timer
    docVec docsToCheck_Copy;
    for (int i = 0; i < docsToCheck.size(); i++) // make a copy of documents to test
    {
        doc currDoc;
        currDoc.title = docsToCheck[i].title;
        currDoc.content = docsToCheck[i].content;
        docsToCheck_Copy.push_back(currDoc);
    }
    naiveBayesClassifyMany(docsToCheck_Copy); // classify the copy set
    ofstream resultsFile; // open file to print results to
    resultsFile.open ("results.txt", fstream::app);


    double correct = 0;
    for (int i = 0; i < docsToCheck.size(); i++) // check how many are accurate
    {
        resultsFile << "Prediction: " << docsToCheck_Copy[i].cat << ", Actual Cat: " << docsToCheck[i].cat << ", Doc: " << docsToCheck_Copy[i].title << " \n ";
        if (docsToCheck_Copy[i].cat == docsToCheck[i].cat)
        {
            correct++;
        }
    }
    double accuracy = (correct / docsToCheck.size()) * 100; // calculate percentage
    cout << "Acccuracy: " << accuracy << "%. NumDocsInSet: " << docsToCheck.size() << endl;
    resultsFile << "\n Acccuracy: " << accuracy << "%." << " \n \n";
    resultsFile.close();

    float secsElapsed3 = (float)(clock() - startTime3)/CLOCKS_PER_SEC;
    cout << "Time for finding accuracy " << secsElapsed3 << " s."<< endl; // timer

}
