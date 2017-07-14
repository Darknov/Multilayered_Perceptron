#pragma once
#include <vector>
#include "Neuron.h"
#include <string>
class Network
{
public:
	Network(std::vector<int> numberOfNeuronsInEachLayer, bool isBias);
	Network(std::string sciezkaDoPliku);
	~Network();
	void tellAboutYourself(double min, double max, double ra, double rb);
	void setWeights();
	void setEveryNeuronOutput();
	void setInNeuronsOutput(std::vector<double> trainingOutputs); //basically propagate forward
	void propagateBackward(std::vector<double> trainingOutPatterns);
	void calculateGlobalError(std::vector<double> trainingOutPattern);// dla danego wzorca, tylko do uzycia gdy byl pokazany wzorzec ktory mu odpowiada
	void calculateGlobalError(std::vector<std::vector<double>> trainingInPatterns, std::vector<std::vector<double>> trainingOutPatterns);// dla wszystkich wzorcow

	double sigmoid(double z);
	double sigmoida(double z);
	double sigmoidDerivative(double z);
	double sigmoidDerivativea(double z);

	bool saveToFile(std::string sciezkaDoPliku);

	double learningSpeed;
	double momentum;
	double isBias;
	double globalError;

	std::vector<std::vector<Neuron>> neurons;


};

