#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <list>
#include <array>
#include <iostream>
#include <algorithm>
#include "ExperimentManager.h"

using namespace std;


const string SIZE_OPERATOR = "-s";
const string EXECUTION_NUMBER_OPERATOR = "-e";
const string CONCURRENCY_OPERATOR = "-c";

void printHelp()
{

	cout << endl << "blasBenchs <blas function 1> <blas function 2> -c -s <input size> -e <execution number> " << endl;
	cout << "functions: ";

	for( string function : ExperimentManager::DISPONIBLE_FUNCTIONS)
	{
		cout<< function << " ";
	}
	cout << endl;

	cout << SIZE_OPERATOR <<" \t sets input matrix size (all matrixes used are squared, 512 by standard)" << endl << endl;
	cout << EXECUTION_NUMBER_OPERATOR <<" \t sets the number of times that each cublas or cusolver function will be called (10 by standard)" << endl << endl;
	cout << CONCURRENCY_OPERATOR <<"\t use streams and execute concurrently" << endl;
	cout << "for memory usage economy both functions will be using the same input matrices when possible" <<endl;

}

bool checkParamFunctions(char *argv[], int func1pos, int func2pos)
{

	auto firstArg = find(ExperimentManager::DISPONIBLE_FUNCTIONS.begin(),ExperimentManager::DISPONIBLE_FUNCTIONS.end(),string(argv[func1pos]));
	auto secondArg = find(ExperimentManager::DISPONIBLE_FUNCTIONS.begin(),ExperimentManager::DISPONIBLE_FUNCTIONS.end(),string(argv[func2pos]));

	if(firstArg == ExperimentManager::DISPONIBLE_FUNCTIONS.end())
	{
		cout << "unknown parameter " << string(argv[func1pos]) << endl;
		printHelp();
		return false;
	}

	if(secondArg == ExperimentManager::DISPONIBLE_FUNCTIONS.end())
	{
		cout << "unknown parameter " << string(argv[func2pos]) << endl;
		printHelp();
		return false;
	}

	return true;

}

int main(int argc, char *argv[])
{
	bool concurrent = false;
	int inputSize = ExperimentManager::STANDARD_SIZE;
	int attempts = ExperimentManager::STANDARD_ATTEMPTS;
	string func1;
	string func2;

	if(argc == 2 && string(argv[1]) == "-h")
		printHelp();
	else{

		if(checkParamFunctions(argv,1,2)){
			func1 = string(argv[1]);
			func2 = string(argv[2]);
		}
		else{
			printHelp();
			return 0;
		}
		for(int i = 3; i < argc; ++i){
			string arg(argv[i]);
			if(arg == CONCURRENCY_OPERATOR)
				concurrent = true;
			else if(arg == SIZE_OPERATOR )
			{
				if(i+1 < argc)
				{
					arg = string(argv[i+1]);
					inputSize = stoi(arg);
					i++;
				}
				else
				{
					cout << "invalid argument" << endl;
					printHelp();
					return 0;

				}

			}
			else if(arg == EXECUTION_NUMBER_OPERATOR){
				if(i+1 < argc)
				{
					attempts = stoi(arg);
					i++;
				}
				else
				{
					cout << "invalid argument" << endl;
					printHelp();
					return 0;

				}
			}
			else{
				cout << "invalid argument" << endl;
				printHelp();
				return 0;
			}
		}
	}

	  ExperimentManager manager(inputSize,attempts,concurrent, func1, func2 );
	  manager.runExperiments();




}


