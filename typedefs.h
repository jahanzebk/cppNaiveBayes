#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <iostream>
#include <vector>
#include <map>

using namespace std;

typedef vector<string> svec;
typedef vector<svec> smatrix;
typedef vector<int> ivec;
typedef vector<double> dvec;

typedef map<string, int> mapSI;
typedef map<string, double> mapSD;
typedef vector<mapSI> mapSIVec;
typedef vector<mapSD> mapSDVec;


typedef map<string, mapSD> mapSmapSD;


//operator overload for outputting svec
ostream& operator << (ostream &stream, svec &obj);

//operator overload for outputting dvec
ostream& operator << (ostream &stream, dvec &obj);

#endif
