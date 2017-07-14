#pragma once
#include <vector>
#include <cmath>
class Neuron
{
public:
	Neuron();
	~Neuron();


	std::vector<double> W;
	std::vector<double> previousDeltaWeights;
	double gradient;
	double output;
};

