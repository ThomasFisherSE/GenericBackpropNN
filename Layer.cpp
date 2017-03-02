#include "Layer.h"
#include <math.h>
#include <assert.h>


Layer::Layer()
{
}


Layer::~Layer()
{

}

vector<double>& Layer::propagateWeigths(vector<double> input)
{

}

vector<double>& Layer::backPropagate(vector<double> input)
{
	
	/* 1: Initialize all weights (w_ij)^l at random */
	m_weights.randn();

	/* 2 : for t = 0, 1, 2, . . . do */
	for (int i = 0; i < 10000; i++) {
		/* 3 : Pick n ∈{ 1, 2, · · · , N } */
		int n = rand();

		/* 4 :	Forward : Compute all (x_j)^l */
		// (x_j)^l = θ(sum of{(w_ij)^l)((x_i)^(l-1))})

		/* 5 :	Backward : Compute all (δ_j)^l */
		// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L

		// (δ_i)^(l-1) = (1 - (((x_i)^(l-1))^2)(sum of{((w_ij)^l)((δ_j)^l)}


		/* 6 :	Update the weights : (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l */


		/* 7:	Iterate to the next step until it is time to stop */
	}

	/* 8 : Return the final weights (w_ij)^l */
	
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