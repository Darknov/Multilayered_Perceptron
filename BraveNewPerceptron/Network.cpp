#include "Network.h"
#include "Neuron.h"
#include "Math.h"
#include <iostream>
#include "Random.h"
#include <string>
#include <fstream>

using namespace std;


Network::Network(std::vector<int> numberOfNeuronsInEachLayer, bool isBias)
{
	this->isBias = isBias;
	for (int i = 0; i < numberOfNeuronsInEachLayer.size(); i++)
	{

		this->neurons.push_back(vector<Neuron>());
		for (int j = 0; j < numberOfNeuronsInEachLayer[i]; j++)
		{
			this->neurons[i].push_back(Neuron());
		}
	}

}

Network::Network(string sciezkaDoPliku)
{
	ifstream myfile;
	vector<string> lines = vector<string>();
	myfile.open(sciezkaDoPliku);
	myfile >> this->learningSpeed;
	myfile >> this->momentum;
	myfile >> this->isBias;
	int numberOfLayers;
	myfile >> numberOfLayers;
	for (int i = 0; i < numberOfLayers; i++)
	{
		this->neurons.push_back(vector<Neuron>());
	}
	
	for (int i = 0; i < this->neurons.size(); i++)
	{
		int numberOfNeurons;
		myfile >> numberOfNeurons;
		for (int j = 0; j < numberOfNeurons; j++)
		{
			this->neurons[i].push_back(Neuron());
		}
	}

	for (int i = 0; i < this->neurons[0].size(); i++)
	{
		this->neurons[0][i].W.push_back(1.0);
	}
	int biasCounter = 0;
	if (isBias)
		biasCounter = 1;
	for (int i = 1; i < this->neurons.size(); i++)
	{
		for (int j = 0; j < this->neurons[i].size(); j++)
		{
			for (int k = 0; k < neurons[i - 1].size() + biasCounter; k++)
			{
				double wage;
				myfile >> wage;
				this->neurons[i][j].W.push_back(wage);
				this->neurons[i][j].previousDeltaWeights.push_back(0);
			}
		}
	}
	
	myfile.close();


}


Network::~Network()
{
}

void Network::tellAboutYourself(double min, double max, double ra, double rb)
{
	/*
	cout << "I have: " << endl;
	for (int i = 0; i < this->neurons.size(); i++)
	{
		cout << "layer " << i + 1 << ": " << neurons[i].size() << " neurons" << endl;
		for (int j = 0; j < this->neurons[i].size(); j++)
		{
			cout << "Neuron: " << j << ", Weights: ";
			for (int k = 0; k < this->neurons[i][j].W.size() - 1; k++)
			{
				cout << this->neurons[i][j].W[k] << ", ";
			}
			if(isBias)
				cout << this->neurons[i][j].W.back() << ", ";
			cout << endl;
			cout << "output: " << this->neurons[i][j].output;
			cout << endl;
		}
	}
	*/
	for (int i = 0; i < this->neurons.back().size(); i++)
	{
		cout << "expected output: " << ((this->neurons[0][i].output - rb)*(max - min) / (ra - -rb) + min);
		cout << ", output: " << ((this->neurons.back()[i].output - rb)*(max - min) / (ra - -rb) + min);
		cout << endl;
	}

	cout << endl;
	cout << "Learning Speed: " << this->learningSpeed << endl;
	cout << "Momentum: " << this->momentum << endl;
	

}

void Network::setWeights()
{
	Random myRandom = Random();
	for (int i = 0; i < this->neurons[0].size(); i++)
	{
		this->neurons[0][i].W.push_back(1.0);
	}

	for (int i = 1; i < this->neurons.size(); i++)
	{
		for (int j = 0; j < this->neurons[i].size(); j++)
		{
			for (int k = 0; k < neurons[i-1].size() + 1; k++)
			{
				this->neurons[i][j].W.push_back( myRandom.nextDoubleExcludingTop(-1.0, 1.0) );
				this->neurons[i][j].previousDeltaWeights.push_back(0);
			}
		}
	}

}

void Network::setEveryNeuronOutput() //basically propagate forward
{
	double sum = 0;
	for (int i = 1; i < this->neurons.size(); i++)
	{
		for (int j = 0; j < this->neurons[i].size(); j++)
		{
			sum = 0;
			for (int k = 0; k < this->neurons[i][j].W.size() - 1; k++)
			{
				sum += this->neurons[i][j].W[k] * this->neurons[i - 1][k].output;

				
			}

			if (this->isBias)
			{
				sum += this->neurons[i][j].W.back() * 1;
			}

			this->neurons[i][j].output = this->sigmoida(sum);
		}
	}
}

void Network::setInNeuronsOutput(vector<double> trainingOutputs)
{
	for (int i = 0; i < neurons[0].size(); i++)
	{
		neurons[0][i].output = trainingOutputs[i];
	}
}

void Network::propagateBackward(vector<double> trainingOutPatterns)
{

	for (int m = 0; m < neurons.back().size(); m++)
	{
		neurons.back()[m].gradient = ((trainingOutPatterns[m] - neurons.back()[m].output)
			* this->sigmoidDerivativea(neurons.back()[m].output));

		double delta = 0;
		for (int n = 0; n < neurons.back()[m].W.size() - 1; n++)
		{
			delta = this->learningSpeed*(neurons.back()[m].gradient * (neurons[neurons.size() - 2][n].output)) 
				+ this->momentum*neurons.back()[m].previousDeltaWeights[n];
			neurons.back()[m].W[n] = neurons.back()[m].W[n]	+ delta;
			neurons.back()[m].previousDeltaWeights[n] = delta;
		}
		if (this->isBias)
		{
			delta = this->learningSpeed*(neurons.back()[m].gradient * (1)) + this->momentum*neurons.back()[m].previousDeltaWeights.back();
			neurons.back()[m].W.back() = neurons.back()[m].W.back() + delta;
			neurons.back()[m].previousDeltaWeights.back() = delta;
		}
	}

	for (int l = neurons.size() - 2; l > 0 ; l--)
	{
		for (int m = 0; m < neurons[l].size(); m++)
		{
			double sum = 0;
			for (int j = 0; j < neurons[l + 1].size(); j++)
			{
				sum = sum + (neurons[l + 1][j].gradient * neurons[l + 1][j].W[m]);
			}

			neurons[l][m].gradient = sum * sigmoidDerivativea(neurons[l][m].output);
		}
	}

	for (int l = neurons.size() - 2; l > 0; l--)
	{
		for (int m = 0; m < neurons[l].size(); m++)
		{
			double delta = 0;
			for (int w = 0; w < neurons[l][m].W.size() - 1; w++)
			{
				delta = this->learningSpeed*(neurons[l][m].gradient * neurons[l - 1][w].output) + this->momentum*neurons[l][m].previousDeltaWeights[w];
				neurons[l][m].W[w] = neurons[l][m].W[w] + delta;
				neurons[l][m].previousDeltaWeights[w] = delta;
			}
			if (this->isBias)
			{
				delta = this->learningSpeed*(neurons[l][m].gradient * 1) + this->momentum*neurons[l][m].previousDeltaWeights.back();
				neurons[l][m].W.back() = neurons[l][m].W.back() + delta;
				neurons[l][m].previousDeltaWeights.back() = delta;
			}

		}
	}
}




void Network::calculateGlobalError(vector<double> trainingOutPatterns)
{
	double error = 0;
	for (int i = 0; i < neurons.back().size(); i++)
	{
		double temp = (neurons.back()[i].output - trainingOutPatterns[i]);
		error += temp * temp;
	}
	this->globalError = error/2.0;
}

void Network::calculateGlobalError(std::vector<std::vector<double>> trainingInPatterns, std::vector<std::vector<double>> trainingOutPatterns)
{
	double error = 0;
	for (int i = 0; i < trainingInPatterns.size(); i++)
	{
		double localError = 0;
		this->setInNeuronsOutput(trainingInPatterns[i]);
		this->setEveryNeuronOutput();
		for (int j = 0; j < neurons.back().size(); j++)
		{
			double temp = (neurons.back()[j].output - trainingOutPatterns[i][j]);
			localError += temp * temp;
		}
		error += (localError / 2.0);
	}
	this->globalError = error/(double)trainingInPatterns.size();
}

bool Network::saveToFile(string sciezkaDoPliku)
{
	ofstream myfile;
	myfile.open(sciezkaDoPliku);
	myfile << this->learningSpeed <<"\n";
	myfile << this->momentum << "\n";
	myfile << this->isBias << "\n";
	myfile << this->neurons.size() << "\n";
	for (int i = 0; i < neurons.size(); i++)
	{
		myfile << this->neurons[i].size() << " ";
	}
	myfile << "\n\n";

	for (int i = 1; i < neurons.size(); i++)
	{
		for (int j = 0; j < neurons[i].size(); j++)
		{
			for (int k = 0; k < neurons[i][j].W.size(); k++)
			{
				myfile << this->neurons[i][j].W[k] << " ";
			}
			myfile << "\n";
		}
		myfile << "\n";
	}

	for (int i = 1; i < neurons.size(); i++)
	{
		for (int j = 0; j < neurons[i].size(); j++)
		{
			for (int k = 0; k < neurons[i][j].W.size(); k++)
			{
				myfile << this->neurons[i][j].previousDeltaWeights[k] << " ";
			}
			myfile << "\n";
		}
		myfile << "\n";
	}

	myfile.close();

	return true;
}


double Network::sigmoid(double z)
{
	return 1.0 / (1.0 + pow(2.72, -1 * z));
}

double Network::sigmoida(double z)
{
	return (1.0 / (1.0 + exp(-z)));
	//return (22.0 / (1.0 + exp(-z)));
	//return z;
}

double Network::sigmoidDerivative(double z)
{
	return pow(2.72, z) / ((2.72 + 1)*(2.72 + 1));
}

double Network::sigmoidDerivativea(double z)
{
	return 1.0*(z * (1.0 - z));
	//return 22.00*(z * (1.0 - z));
	//return 1;
}