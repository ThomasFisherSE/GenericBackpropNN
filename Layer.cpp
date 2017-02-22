#include "Layer.h"
#include <math.h>
#include <assert.h>


Layer::Layer()
{
}


Layer::~Layer()
{
	/*
	if (neurons)
	{
		for (int i = 0; i < neuronCount; i++)
		{
			delete neurons[i];
		}
		delete[] neurons;
	}
	if (layerInput)
	{
		delete[] layerInput;
	}
	*/
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