#include <iostream>
#include <vector>
#include "Network.h"
#include <algorithm>
#include <fstream>
#include "gnuplot_i.hpp"
using namespace std;

vector<vector<double>> loadPatternsFromFile(string path)
{
	vector<vector<double>> patterns = vector<vector<double>>();
	ifstream file;
	file.open(path);
	int amountOfPatterns;
	int amountOfNumbers;
	file >> amountOfPatterns;
	file >> amountOfNumbers;
	for (int i = 0; i < amountOfPatterns; i++)
	{
		patterns.push_back(vector<double>());
		for (int j = 0; j < amountOfNumbers; j++)
		{
			int number;
			file >> number;
			patterns[i].push_back(number);
			cout << number << " ";
		}
		cout << endl;
	}


	return patterns;
}

vector<vector<double>> loadPatternsFromFileCustom1(string path, double &ra, double &rb, double &min, double &max)
{
	vector<vector<double>> patterns = vector<vector<double>>();
	ifstream file;
	file.open(path);
	double tempTab[7];
	int temp;
	int numberOfPattern = 0;
	max = 0;
	min = 30;
	while (file >> tempTab[0] >> tempTab[1] >> tempTab[2] >> tempTab[3] >> tempTab[4] >> tempTab[5] >> tempTab[6] >> temp)
	{
		patterns.push_back(vector<double>());
		for (int i = 0; i < 7; i++)
		{
			if (tempTab[i] > max)
			{
				max = tempTab[i];
			}

			if (tempTab[i] < min)
			{
				min = tempTab[i];
			}
			cout << tempTab[i] << " ";
			patterns[numberOfPattern].push_back(tempTab[i]);
		}
		cout << endl;
		numberOfPattern++;
	}
	
	double a = min;
	double b = max;
	//double ra = 1.0;
	//double rb = 0.0;

	for (int i = 0; i < patterns.size(); i++)
	{
		for (int j = 0; j < patterns[i].size(); j++)
		{
			patterns[i][j] = (((ra - rb) * (patterns[i][j] - a)) / (b - a)) + rb;
		}
	}
	
	return patterns;
}

void saveError(string path, double error, int epoch)
{
	ofstream file;
	file.open(path, std::ios::app);
	file << epoch << " " << error << "\n";
	file.close();
}

int main()
{
	int toStopFromClosing;
	vector<double> trainingPatternIn1 = { 1.0, 0.0, 0.0, 0.0 },
				   trainingPatternIn2 = { 0.0, 1.0, 0.0, 0.0 },
				   trainingPatternIn3 = { 0.0 ,0.0, 1.0, 0.0 },
				   trainingPatternIn4 = { 0.0, 0.0, 0.0, 1.0 };
	vector<vector<double>> trainingPatternsIn =
					{ trainingPatternIn1,
					  trainingPatternIn2,
					  trainingPatternIn3,
					  trainingPatternIn4 };

	vector<double> trainingPatternOut1 = { 1.0, 0.0, 0.0, 0.0 },
				   trainingPatternOut2 = { 0.0, 1.0, 0.0, 0.0 },
				   trainingPatternOut3 = { 0.0 ,0.0, 1.0, 0.0 },
				   trainingPatternOut4 = { 0.0, 0.0, 0.0, 1.0 };
	vector<vector<double>> trainingPatternsOut =
					{ trainingPatternOut1,
					  trainingPatternOut2,
					  trainingPatternOut3,
					  trainingPatternOut4 };

	// wczytywanie patternsow z pliku
	//trainingPatternsIn = loadPatternsFromFile("inPatterns");
	//trainingPatternsOut = loadPatternsFromFile("inPatterns");
	//trainingPatternsIn = loadPatternsFromFileCustom1("seeds_dataset1.txt");
	//trainingPatternsOut = loadPatternsFromFileCustom1("seeds_dataset1.txt");

	double ra = 1.0;
	double rb = 0.0;
	double min;
	double max;

	trainingPatternsIn = loadPatternsFromFileCustom1("seeds_dataset.txt",ra,rb,min,max);
	trainingPatternsOut = loadPatternsFromFileCustom1("seeds_dataset.txt", ra, rb, min, max);
	string nazwaZadania = "proba4";


	vector<int> utilTrainingPatterns = vector<int>();
	for (int i = 0; i < trainingPatternsIn.size(); i++)
	{
		utilTrainingPatterns.push_back(i);
	}

	vector<int> numberOfNeuronsInEachLayer = { 7,6,7 };

	double learingSpeed = 0.9;
	double momentum = 0.6;
	bool isBias = true;
	

	// ustawienia uczenia
	int maxInterations = 1000; //ilosc iteracji
	int interationsDisplay = 10; // co ile iteracji pokazujemy stan sieci
	double desirableError = 0.000001; // pozadany globalError
	bool isShufflingPatterns = true; // pokazywanie patternow w kolejnosci losowej
	bool isSavingGlobalError = true; // automatyczne zapisywanie kolejnych bledow sieci 
	bool isSavingNetwork = true; // automatyczne zapisanie sieci po nauczeniu
	bool IS_LEARNING = true; // testowanie czy uczenie

	vector<int> epoki;
	vector<double> blad;

	// tworzenie sieci 

	// podstawowe tworzenie sieci
	Network myNetwork = Network(numberOfNeuronsInEachLayer, isBias);
	myNetwork.setWeights();
	myNetwork.learningSpeed = learingSpeed;
	myNetwork.momentum = momentum;

	// wczytanie sieci z pliku
	//Network myNetwork = Network("siec1"); 


	if (IS_LEARNING)
	{
		for (int i = 0; i < maxInterations + 1; i++)
		{
			if (i % (interationsDisplay) == 0)
			{
				cout << "------------------------------" << endl;
				cout << "EPOCH: " << i  << endl;
				cout << "------------------------------" << endl;
			}
			if (isShufflingPatterns)
			{
				random_shuffle(utilTrainingPatterns.begin(), utilTrainingPatterns.end());
			}
			for (int j = 0; j < trainingPatternsIn.size(); j++)
			{
				int patternNumber = utilTrainingPatterns[j];
				if (i % (interationsDisplay) == 0)
				{
					cout << patternNumber << " training pattern: ";
					for (int t = 0; t < 4; t++)
						cout << trainingPatternsIn[patternNumber][t] << " ";
					cout << endl;
				}

				myNetwork.setInNeuronsOutput(trainingPatternsIn[patternNumber]);
				myNetwork.setEveryNeuronOutput();
				myNetwork.propagateBackward(trainingPatternsOut[patternNumber]);
				if (i % (interationsDisplay) == 0)
				{
					myNetwork.tellAboutYourself(min, max, ra, rb);
					cout << endl;
				}
			}
			if (i % (interationsDisplay) == 0)
			{
				myNetwork.calculateGlobalError(trainingPatternsIn, trainingPatternsOut);
				saveError("error" + nazwaZadania, myNetwork.globalError, i);
				epoki.push_back(i);
				blad.push_back(myNetwork.globalError);
				cout << "Global error: " << myNetwork.globalError << endl;
			}


			if (myNetwork.globalError < desirableError)
			{
				break;
			}
		}
		// jesli chcesz zapisac siec
		if (isSavingNetwork)
		{
			myNetwork.saveToFile("siec" + nazwaZadania);
		}
	}
	else {
		for (int j = 0; j < trainingPatternsIn.size(); j++)
		{
			cout << "training pattern: ";
			for (int t = 0; t < 4; t++)
			{
				cout << trainingPatternsIn[j][t] << " ";
			}
			cout << endl;
			myNetwork.setInNeuronsOutput(trainingPatternsIn[j]);
			myNetwork.setEveryNeuronOutput();
			myNetwork.tellAboutYourself(min, max, ra, rb);
			myNetwork.calculateGlobalError(trainingPatternsIn, trainingPatternsOut);
			cout << "Global error: " << myNetwork.globalError << endl;
			cout << endl;
		}
	}

	try {
		// Create Gnuplot object

		Gnuplot gp;
		Gnuplot::set_GNUPlotPath("c:\\gnuplot\\");
		// Configure the plot

		gp.set_style("lines");
		gp.set_xlabel("epoki");
		gp.set_ylabel("blad");

		// Plot the data

		gp.plot_xy(epoki, blad);

		cout << "Press Enter to quit...";
		cin.get();
	}
	catch (const GnuplotException& error) {

		// Something went wrong, so display error message

		cerr << error.what() << endl;
	}
	
	


	cin >> toStopFromClosing;
	

}