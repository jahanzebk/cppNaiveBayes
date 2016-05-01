#ifndef WEIGHT_H
#define WEIGHT_H

#include <vector>

using namespace std;

// This class holds data for data related to each weight for each word with respect to each category
// It describes the importance of a word to a category, the lower the weight, the more important
class weight
{
    public:
        string cat;
        string word;
        double weight;
        // this is for extensibility
};

typedef vector<weight> wVec;

#endif
