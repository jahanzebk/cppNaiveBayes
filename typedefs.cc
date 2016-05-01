#include "typedefs.h"
#include <iostream>
#include <vector>

using namespace std;

//operator overload for outputting svec
ostream& operator << (ostream &stream, svec &obj)
{
    for (int i =0; i < obj.size(); i++)
    {
        stream << obj[i] << ", ";
    }
    return stream;
}

//operator overload for outputting dvec
ostream& operator << (ostream &stream, dvec &obj)
{
    for (int i =0; i < obj.size(); i++)
    {
        stream << obj[i] << ", ";
    }
    return stream;
}