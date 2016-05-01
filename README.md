# cppNaiveBayes
A (Term Weight Complement) Naive Bayes implementation in C++.

## Introduction
This code, written by Jahanzeb Khan, is an implementation of the (term-weighted) 
Complement Naive Bayes Algorithm in C++. The paper that describes the algorithm,
and which was followed to write this code is:
https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

## Compilation
On Ubuntu atleast, (make sure to have boost installed)
g++ main.cc NaiveBayes.cc typedefs.cc -lboost_filesystem -lboost_system -o main
or
g++ -L /usr/include/boost main.cc ...(rest is same as above)

Confirmed working with Boost 1.54.

