#ifndef DOC_H
#define DOC_H

#include <vector>

using namespace std;

// this class holds data related to each document, regardless of what the document will be used for (testing or training)
class doc
{
    public:
        string cat;
        string content;
        string title;
        // this is for extensibility
};

typedef vector<doc> docVec;

#endif
