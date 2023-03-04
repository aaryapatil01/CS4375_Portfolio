// Names: Austin Girouard and Aarya Patil
// Project : Portfolio Assignment 3 (ML Algorithms From Scratch)
// Date : 03/04/23

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#define PI 3.141592653589793238462643383279502
#include <cmath>

using namespace std;

double calcAgeLikelihood(double, double, double);
double calcAccuracy(vector<int>, vector<int>, int, int);
double calcSensitivity(vector<int>, vector<int>, int, int);
double calcSpecificity(vector<int>, vector<int>, int, int);

int main(int argc, char** argv)
{
    ifstream input;
    string line;
    string pclass_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1200;
    vector<int> pclass(MAX_LEN);
    vector<int> survived(MAX_LEN);
    vector<int> sex(MAX_LEN);
    vector<float> age(MAX_LEN);
    int trainSize = 800;
    int testSize;


    string fileName = "titanic_project.csv";

    cout << "Opening " + fileName + " file." << endl;

    input.open(fileName);
    if (!input.is_open()) {
        cout << "Could not open file " + fileName + "." << endl;
        return 1;
    }

    // Read in header line from CSV
    cout << "Reading line 1" << endl;
    getline(input, line);

    int numDataPoints = 0;
    // Read in all data points from csv file
    while (input.good()) {
        // Read in inputs as string
        getline(input, pclass_in, ','); // Disregard first data element
        getline(input, pclass_in, ',');
        getline(input, survived_in, ',');
        getline(input, sex_in, ',');
        getline(input, age_in, '\n');

        // Convert string to int, store in vectors
        pclass[numDataPoints] = stoi(pclass_in);
        survived[numDataPoints] = stoi(survived_in);
        sex[numDataPoints] = stoi(sex_in);
        // Convert age to float, store in vector
        age[numDataPoints] = stof(age_in);

        numDataPoints++;
    }

    // Close input file
    input.close();

    // Calculate testing data set size
    testSize = numDataPoints - trainSize;

    // Initialize metrics for Naive Bayes
    double apriori[2] = { 0.0 };     // [0] = died, [1] = survived
    double pclassCP[2][3] = { 0.0 }; // Row[0] = died, Row[1] = survived, Col[x] = pclass
    double sexCP[2][2] = {0.0};      // Row[0] = died, Row[1] = survived, Col[0] = female, Col[1] = male
    double ageMean[2] = { 0.0 };     // [0] = death mean age, [1] = survived mean age
    double ageVariance[2] = { 0.0 }; // [0] = death variance age, [1] = survived variance age
    int survivedCount = 0;
    int deathCount = 0;


    // Store start time for data training
    auto startTime = chrono::steady_clock::now();

    // Calculate suvival and death counts
    for (int i = 0; i < trainSize; i++)
    {
        if (survived[i] == 1)
            survivedCount++;
        else
            deathCount++;
    }

    // Calculate apriori values
    apriori[0] = (double)deathCount / trainSize;
    apriori[1] = (double)survivedCount / trainSize;

    // Calculate conditional probabilities and the mean ages for survival and death
    // First, count all instances
    for (int i = 0; i < trainSize; i++)
    {
        // pclass starts at 1, subtract 1 to adjust for array indexing
        pclassCP[survived[i]][pclass[i] - 1] += 1;
        sexCP[survived[i]][sex[i]] += 1;
        ageMean[survived[i]] += age[i];
    }

    // Second, divide by totals for survival and death to get probabilities and means
    ageMean[0] /= deathCount;    // Calculate mean death age
    ageMean[1] /= survivedCount; // Calculate mean survival age

    // Calculate probabilities of survival and death for each pclass
    for (int i = 0; i < 3; i++)
    {
        pclassCP[0][i] /= deathCount;
        pclassCP[1][i] /= survivedCount;
    }

    // Calculate probabilities of survival and death for each sex
    for (int i = 0; i < 2; i++)
    {
        sexCP[0][i] /= deathCount;
        sexCP[1][i] /= survivedCount;
    }

    // Calculate age variance
    for (int i = 0; i < trainSize; i++)
        ageVariance[survived[i]] += pow((age[i] - ageMean[survived[i]]), 2);
    
    ageVariance[0] /= deathCount - 1;
    ageVariance[1] /= survivedCount - 1;

    // Store end time for data training
    auto endTime = chrono::steady_clock::now();


    // Initialize vectors to hold predictions
    vector<int> predictions(testSize);

    // Test predictions
    for (int i = trainSize; i < numDataPoints; i++)
    {
        // Calculate probability of death and survival using formulas from rmd file
        double probDeath = pclassCP[0][pclass[i] - 1] * sexCP[0][sex[i]] * apriori[0] * calcAgeLikelihood(age[i], ageMean[0], ageVariance[0]);
        double probSurvive = pclassCP[1][pclass[i] - 1] * sexCP[1][sex[i]] * apriori[1] * calcAgeLikelihood(age[i], ageMean[1], ageVariance[1]);
        double denom = probDeath + probSurvive;

        double math = probSurvive / denom;
        predictions[i - trainSize] = (probSurvive / denom) > 0.5 ? 1 : 0;
    }


    // Print results
    cout << "Accuracy: " << calcAccuracy(survived, predictions, trainSize, numDataPoints) << endl;
    cout << "Sensitivity: " << calcSensitivity(survived, predictions, trainSize, numDataPoints) << endl;
    cout << "Specificity: " << calcSpecificity(survived, predictions, trainSize, numDataPoints) << endl;
    cout << "Running time of training the data: " << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() << "ms" << endl;
}


/*******************************************************************************
* This function calculates the age likelihood given an age, mean, and variance *
* Input: double age, double mean, double variance                              *
* Output: Age Likelihood                                                       *
********************************************************************************/
double calcAgeLikelihood(double age, double mean, double variance)
{
    return 1.0 / sqrt(2 * PI * variance) * exp(-((age - mean) * (age - mean)) / (2 * variance));
}


/********************************************************************************************
* This function calculates the accuracy of predictions for survival.                        *
* Input: vector<int> with correct survival data, vector<int> with predictions of survival,  *
*   int for the starting index of the testing data, int for the ending index of the testing *
*   data.                                                                                   *
* Output: Double accuracy                                                                   *
*********************************************************************************************/
double calcAccuracy(vector<int> survived, vector<int> predictions, int startIndex, int endIndex)
{
    int FP = 0, FN = 0, TP = 0, TN = 0;
    for (int i = startIndex; i < endIndex; i++)
    {
        if (survived.at(i) == 0 && predictions[i - startIndex] == 1)
            FP++;
        if (survived.at(i) == 1 && predictions[i - startIndex] == 0)
            FN++;
        if (survived.at(i) == 1 && predictions[i - startIndex] == 1)
            TP++;
        if (survived.at(i) == 0 && predictions[i - startIndex] == 0)
            TN++;
    }

    return (0.0 + TP + TN) / (endIndex - startIndex);
}


/********************************************************************************************
* This function calculates the sensitivity of predictions for survival.                     *
* Input: vector<int> with correct survival data, vector<int> with predictions of survival,  *
*   int for the starting index of the testing data, int for the ending index of the testing *
*   data.                                                                                   *
* Output: Double sensitivity                                                                *
*********************************************************************************************/
double calcSensitivity(vector<int> survived, vector<int> predictions, int startIndex, int endIndex)
{
    int FP = 0, FN = 0, TP = 0, TN = 0;
    for (int i = startIndex; i < endIndex; i++)
    {
        if (survived.at(i) == 0 && predictions[i - startIndex] == 1)
            FP++;
        if (survived.at(i) == 1 && predictions[i - startIndex] == 0)
            FN++;
        if (survived.at(i) == 1 && predictions[i - startIndex] == 1)
            TP++;
        if (survived.at(i) == 0 && predictions[i - startIndex] == 0)
            TN++;
    }

    return (0.0 + TP) / (0.0 + TP + FN);
}


/********************************************************************************************
* This function calculates the specificity of predictions for survival.                     *
* Input: vector<int> with correct survival data, vector<int> with predictions of survival,  *
*   int for the starting index of the testing data, int for the ending index of the testing *
*   data.                                                                                   *
* Output: Double specificity                                                                *
*********************************************************************************************/
double calcSpecificity(vector<int> survived, vector<int> predictions, int startIndex, int endIndex)
{
    int FP = 0, FN = 0, TP = 0, TN = 0;
    for (int i = startIndex; i < endIndex; i++)
    {
        if (survived.at(i) == 0 && predictions[i - startIndex] == 1)
            FP++;
        if (survived.at(i) == 1 && predictions[i - startIndex] == 0)
            FN++;
        if (survived.at(i) == 1 && predictions[i - startIndex] == 1)
            TP++;
        if (survived.at(i) == 0 && predictions[i - startIndex] == 0)
            TN++;
    }

    return (0.0 + TN) / (0.0 + TN + FP);
}