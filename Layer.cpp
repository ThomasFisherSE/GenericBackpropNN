#include "Layer.h"
#include <math.h>
#include <assert.h>


Layer::Layer()
{
}


Layer::~Layer()
{

}

void Layer::initialiseLayer()
{
	m_weights.randn();
}

void Layer::updateWeights()
{
	for (int col = 0; col < m_weights.cols; col++) {
		for (int row = 0; row < m_weights.rows; row++) {
			// New weight = old weight - (eta * x from previous layer * delta for current layer)
			m_weights.at(col, row) -= m_eta * m_prevX * m_delta;
		}
	}
}



vector<double>& Layer::propagateWeigths(vector<vector<double>> input)
{
	// (x_j)^l = θ(sum of{(w_ij)^l)((x_i)^(l-1))})
	//Next x = ThresholdOf(Sum of(Current weight * x from the previous layer))

	
}

vector<double>& Layer::backPropagate(vector<vector<double>> input)
{
	// (δ_i)^(l-1) = (1 - (((x_i)^(l-1))^2)(sum of{((w_ij)^l)((δ_j)^l)}
	// Delta for previous layer = (1 - (x from previous layer squared * (sum of (weights from current layer * next delta))))
	
	
}

/*
void Layer::create(int inputSize, int numberOfNeurons)
{
	assert(inputSize && numberOfNeurons);

	int i;
	neurons = new Neuron*[numberOfNeurons];

	for (i = 0; i < numberOfNeurons; i++)
	{
		neurons[i] = new Neuron;
		neurons[i]->create(inputSize);
	}

	layerInput = new float[inputSize];
	neuronCount = numberOfNeurons;
	inputCount = inputSize;
}
*/

/*
void Layer::calculate()
{
	/*
	int i, j;
	float sum;

	for (i = 0; i < neuronCount; i++)
	{
		sum = 0;
		for (j = 0; j < inputCount; j++)
		{
			sum += neurons[i]->weights[j] * layerInput[j];
		}

		sum += neurons[i]->weightGain *  neurons[i]->gain;

		neurons[i]->output = 1.f / (1.f + exp(-sum));
	}
}
*/