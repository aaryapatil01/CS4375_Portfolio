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

using namespace std;

double sigmoid(double);
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

    // Initialize 2d array for training the data
    double** trainMat = new double* [trainSize];
    for (int i = 0; i < trainSize; i++)
        trainMat[i] = new double[2];

    for (int i = 0; i < trainSize; i++)
    {
        trainMat[i][0] = 1;
        trainMat[i][1] = sex[i];
        //cout << trainMat[i][0] << " " << trainMat[i][1] << endl;
    }

    // Initialize probability and error vectors
    vector<double> prob(trainSize);
    vector<double> error(trainSize);
    // Initialize learning rate
    double learningRate = 0.001;
    // Initialize weights (w0 and w1) to 1
    double w0 = 1;
    double w1 = 1;

    // Store start time for data training
    auto startTime = chrono::steady_clock::now();

    // Train on first 800 data points for 500,000 iterations
    for (int iterate = 0; iterate < 50000; iterate++)
    {
        for (int i = 0; i < trainSize; i++)
        {
            // Logistic Regression based on R code from textbook
            double sigmoidData = sigmoid(trainMat[i][0] * w0 + trainMat[i][1] * w1);
            prob[i] = sigmoidData;
            error[i] = survived[i] - sigmoidData;
            w0 = w0 + learningRate * trainMat[i][0] * error[i];
            w1 = w1 + learningRate * trainMat[i][1] * error[i];
        }
    }


    // Store end time for data training
    auto endTime = chrono::steady_clock::now();

    // Initialize 2d array for testing the data
    //vector<vector<double>> testMat(testSize, vector<double>(2));
    double** testMat = new double* [testSize];
    for (int i = 0; i < testSize; i++)
        testMat[i] = new double[2];

    // Initialize vectors to hold predictions
    vector<int> predictions(testSize);

    for (int i = 0; i < testSize; i++)
    {
        // Initialize testMat values
        testMat[i][0] = 1;
        testMat[i][1] = sex[trainSize + i];
        // Calculate log odds
        double logOdds = testMat[i][0] * w0 + testMat[i][1] * w1;
        // Use log odds to find probability
        double prob = exp(logOdds) / (1 + exp(logOdds));
        predictions[i] = prob > 0.5 ? 1 : 0;
    }

    // Print results
    cout << "Weight coefficients: w0 = " << w0 << ", w1 = " << w1 << endl;
    cout << "Accuracy: " << calcAccuracy(survived, predictions, trainSize, numDataPoints) << endl;
    cout << "Sensitivity: " << calcSensitivity(survived, predictions, trainSize, numDataPoints)  << endl;
    cout << "Specificity: " << calcSpecificity(survived, predictions, trainSize, numDataPoints) << endl;
    cout << "Running time of training the data: " << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() << "ms" << endl;


    // Deallocate memory for arrays
    for (int i = 0; i < testSize; i++)
        delete testMat[i];
    for (int i = 0; i < trainSize; i++)
        delete trainMat[i];
    delete [] testMat;
    delete [] trainMat;
}


/*************************************************************************************
* This function applies a sigmoid function to a given value and returns the result.  *
* Input: Double data to be applied to sigmoid curve                                  *
* Output: Double result of operation                                                 *
**************************************************************************************/
double sigmoid(double z)
{
    return 1.0 / (1 + exp(-z));
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