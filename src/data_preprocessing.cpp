#include<iostream>
#include<fstream>
#include <locale>
#include<sstream>
#include<iomanip>

using namespace std;

string getFilePath(const string& basePath, const int& index, stringstream& ss) 
{
    ss << basePath << setw(3) << setfill('0') << index << ".txt";
    return std::move(ss.str());
}

int main()
{
    ifstream inputFile;

    string dataFilesPaths[5][2] = {
                        {"Bonn-EEG-raw-Dataset/Set_F/F", "Healthy volunteers"},
                        {"Bonn-EEG-raw-Dataset/Set_O/O", "Healthy volunteers"},
                        {"Bonn-EEG-raw-Dataset/Set_N/N", "Epilepsy patients"},
                        {"Bonn-EEG-raw-Dataset/Set_S/S", "Epilepsy patients"},
                        {"Bonn-EEG-raw-Dataset/Set_Z/Z", "Epilepsy patients"}
                        };

    cout << "Starting Data Preprocessing..." << endl;

    stringstream ss;
    string filePath;
    string line;
    string row;

    ofstream outputFile("dataset.csv");
    outputFile << "patient type, seizure type, s1,s2";
    for  (string *path : dataFilesPaths)
    {
        for (int i = 1; i <= 100; i++)
        {
            ss.str("");
            filePath = getFilePath(path[0], i, ss);
            inputFile.open(filePath);

            if (inputFile.is_open())
            {
                row = path[1] + ",";
                while (getline(inputFile, line))
                {
                    
                }

                inputFile.close();
            }
            else
            {
                cerr << "Unable to open file: " << filePath << endl;
                return 1;
            }
        }
    }

    return 0;
}
