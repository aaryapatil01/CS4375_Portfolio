// Name: Aarya Patil
// Project: C++ data exploration (Portfolio)
// Date: 02/01/2023

#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;

//Function to open input file and check for errors
bool openInputFile (ifstream &inputFile, string dataFile) {
    inputFile.open(dataFile);
    
    //test if file opened
    if (inputFile.good())
        return true;
    else {
        cout << "Can't open the input file." << endl;
        cout << "The file name entered was: " << dataFile << endl;
        cout << "Enter another file name to use (or quit): ";
        getline (cin, dataFile);
        if (dataFile != "quit") { //allows user to quit if they want
            cout << "The new file name entered is: " << dataFile << endl;
            return openInputFile(inputFile, dataFile);
        }
        else return false;
    }
}

//Function to calculate the sum of a numeric vector
double calcSum(vector<double> vec) {
    double sum = 0; //declare variable to hold the sum
    
    //Iterate through vector to calculate sum
    for(int i = 0; i < vec.size(); i++){
        sum += vec[i];
    }
    
    return sum;
}

//Function to calculate the mean of a numeric vector
double calcMean(vector<double> vec) {
    double sum = calcSum(vec); //call function to calculate sum
    
    return sum/vec.size(); //return sum divided by total number of elements in vector (mean)
}

//Function to calculate the median of a numeric vector
double calcMedian(vector<double> vec) {
    std::sort(vec.begin(),vec.end()); //sort the vector
    
    return vec.at(vec.size()/2); //return element at center spot of vector
}

//Function to calculate the range of a numeric vector
double calcRange(vector<double> vec) {
    std::sort(vec.begin(),vec.end()); //sort the vector
    
    return vec.at(vec.size()-1) - vec.at(0); //return difference between first and last elem in vector
}

//Function to print out stats for vector provided
void print_stats(vector<double> vec) {
    
    cout << "\tSum: " << calcSum(vec) << endl;
    cout << "\tMean: " << calcMean(vec) << endl;
    cout << "\tMedian: " << calcMedian(vec) << endl;
    cout << "\tRange: " << calcRange(vec) << endl;
}

//Function to calculate covariance between 2 vectors
double covar(vector<double> vec1, vector<double> vec2) {
    //Declare all variables
    double sum = 0;
    double rmMean = calcMean(vec1); //call function to calculate the mean for vec1 (rm)
    double medvMean = calcMean(vec2); //call function to calculate the mean for vec2 (medv)
    
    //Calculate the sum of the product of estimate-actual for each element
    for (int i = 0; i < vec1.size(); i++) {
        sum += (vec1.at(i) - rmMean) * (vec2.at(i) - medvMean);
    }
    
    return sum / (vec1.size() - 1); //return sum divided by total num of elem in vec1 - 1 (covariance)
}

//Function to calculate correlation between 2 vectors
double cor(vector<double> vec1, vector<double> vec2) {
    //Use the covar function to calculate the correlation
    return covar(vec1, vec2) / (sqrt(covar(vec1, vec1)) * sqrt(covar(vec2, vec2)));
}

int main() {
    
    //Define variables
    ifstream inputFile; //ifstream object
    string dataFile = "Boston.csv"; //data file that we want to open
    
    string heading = ""; //holds heading input from file
    string rm_val, medv_val; //the variables that are in the input file
    
    int numObservations = 0; //counter for vector storage and tracker for number of records
    const int MAX_LEN = 1000; //establishes a maximum length for the vectors
    
    vector<double> rm(MAX_LEN); //creates a vector of type double to store all the rm values
    vector<double> medv(MAX_LEN); //creates a vector of type double to store all medv values
    
    cout << "Opening input file: " << dataFile << endl;
    openInputFile(inputFile, dataFile); //call function to open file
    
    //Read information in from the input file
    cout << "\nReading info from file..." << endl;
    getline(inputFile, heading); //read in the heading from the file
    cout << "\tFile heading: " << heading << endl;
    
    while (inputFile.good()) { //while the file still has info to read
        getline(inputFile, rm_val, ','); //reads rm value until it hits a comma (',')
        getline(inputFile, medv_val, '\n'); //reads medv value until it hits a newline ('\n')
        
        rm.at(numObservations) = stof(rm_val); //stores the rm value in the rm vector
        medv.at(numObservations) = stof(medv_val); //stores the medv value in the medv vector
        
        numObservations++;
    }
    
    //Resize both vectors so all the extra space within them is deleted
    rm.resize(numObservations);
    medv.resize(numObservations);
    cout << "\tResized length of vectors rm and medv: " << rm.size() << endl; //print out new size of the vectors
    
    //Close the input file
    cout << "\nClosing input file: " << dataFile << endl;
    inputFile.close();
    
    //Print out all statistics
    cout << "\nSTATISTICS" << endl; //statistics heading
    cout << "Number of records: " << numObservations << endl; //total number of records
    
    cout << "\nStats for rm" << endl; //print stats for rm
    print_stats(rm); //call function to do that
    
    cout << "\nStats for medv" << endl; //print stats for medv
    print_stats(medv); //call function to do that
    
    cout << "\nCovariance = " << covar(rm, medv) << endl; //call function and print out covariance
    
    cout << "\nCorrelation = " << cor(rm, medv) << endl << endl; //call function and print out correlation
    
    return 0;
}
