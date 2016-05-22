# cppNaiveBayes
A (Term Weight Complement) Naive Bayes implementation in C++.

## Introduction
This code, written by Jahanzeb Khan, is an implementation of the (term-weighted) 
Complement Naive Bayes Algorithm in C++. The paper that describes the algorithm,
and which was followed to write this code is:
https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

## Compilation
On Ubuntu (and maybe windows), (make sure to have the C++ boost library installed installed)
g++ main.cc TfidfVectorizer.cc NaiveBayes.cc typedefs.cc -lboost_filesystem -lboost_system -o main
or
g++ -L /usr/include/boost main.cc ...(rest is same as above)

Confirmed working with Boost 1.54.

## Running the Program

### Corpus Structure
When running the program, you need to have a document corpus of text files of the following format.
One folder holding the entire corpus, to be passed as an argument when running the program.
This folder will only have more folders, one for each category, the folder name specifies the category.
Each "category folder" has all the text (currently only .txt supported) files with the article text in them.

For mining such articles and getting text files, another python based project may be uploaded and linked here soon.
