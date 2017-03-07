#include "Network.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Network::Network()
{

}

Network::~Network()
{
	/*
	if (m_hiddenLayers)
	{
		for (int i = 0; i < m_hiddenLayerCount; i++)
		{
			delete m_hiddenLayers[i];
		}

		delete[] m_hiddenLayers;
	}
	*/
}

void Network::learn(vector<vector<double>> data) {
	int numberCorrect = 0;

	/* 1: Initialize all weights (w_ij)^l at random */
	for (int i = 0; i < m_layers.size; i++) {
		m_layers.at(i).initialiseLayer();
	}

	/* 2 : for t = 0, 1, 2, . . . do */
	while (m_recognitionRate < TARGET_RECOGNITION) {
		for (int l = 0; l < m_layers.size; l++) {
			/* 3 : Pick n ∈{ 1, 2, · · · , N } */
			int n = rand();

			/* 4 :	Forward : Compute all (x_j)^l */
			m_layers.at(l).propagateWeigths(data);
		}

		/* 5 :	Backward : Compute all (δ_j)^l */
		for (int l = m_layers.size; l >= 0; l++) {
			// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L
			m_layers.at(l).backPropagate(data);


		}

		/* 6 :	Update the weights : (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l */
		for (int l = 0; l < m_layers.size; l++) {
			m_layers.at(l).updateWeights();
		}


		// Print recognition rate every few iterations
		m_recognitionRate = (numberCorrect / 60000) * 100;
		cout << "Recognition Rate: " << m_recognitionRate << endl;

		/* 7:	Iterate to the next step until it is time to stop */
	}

	/* 8 : Return the final weights (w_ij)^l */


/*
float Layer::learn(const float *desiredOutput, const float *input, float alpha, float momentum)
{
float generalError = 0;
float localError;
float sum = 0, localSum = 0;
float delta, udelta;
float output;

feedforward(input);

int i, j, k;

for (int i = 0; i < m_outputLayer.neuronCount; i++)
{
output = m_outputLayer.neurons[i]->output;
localError = (desiredOutput[i] - output) * output * (1 - output);
generalError += (desiredOutput[i] - output) * (desiredOutput[i] - output);

for (j = 0; j < m_outputLayer.inputCount; j++)
{
delta = m_outputLayer.neurons[i]->deltaValues[j];

udelta = alpha * localError * m_outputLayer.layerInput[j] + delta * momentum;

m_outputLayer.neurons[i]->weights[j] += udelta;
m_outputLayer.neurons[i]->deltaValues[j] = udelta;

sum += m_outputLayer.neurons[i]->weights[j] * localError;
}

m_outputLayer.neurons[i]->weightGain += alpha * localError * m_outputLayer.neurons[i]->gain);

}

for (i = (m_hiddenLayerCount - 1); i >= 0; i--)
{
for (j = 0; j < m_hiddenLayers[i]->neuronCount; j++)
{
output = m_hiddenLayers[i]->neurons[j]->output;

localError = output * (1 - output) * sum;

for (k = 0; k < m_hiddenLayers[i]->inputCount; k++)
{
delta = m_hiddenLayers[i]->neurons[j]->deltaValues[k];
udelta = alpha * localError * m_hiddenLayers[i]->layerInput[k];
m_hiddenLayers[i]->neurons[j]->weights[k] += udelta;
m_hiddenLayers[i]->neurons[j]->deltaValues[k] = udelta;
localSum += m_hiddenLayers[i]->neurons[j]->weights[k] * localError;
}

m_hiddenLayers[i]->neurons[j]->weightGain += alpha * localError * m_hiddenLayers[i]->neurons[j]->gain;
}

sum = localSum;
localSum = 0;
}

for (i = 0; i < m_inputLayer.neuronCount; i++)
{
output = m_inputLayer.neurons[i]->output;
localError = output * (1 - output) * sum;

for (j = 0; j < m_inputLayer.inputCount; j++)
{
delta = m_inputLayer.neurons[i]->deltaValues[j];
udelta = alpha * localError * m_inputLayer.layerInput[j] + delta * momentum;

m_inputLayer.neurons[i]->weights[j] += udelta;
m_inputLayer.neurons[i]->deltaValues[j] = udelta;
}

m_inputLayer.neurons[i]->weightGain += alpha * localError * m_inputLayer.neurons[i]->gain;
}

return generalError / 2;
}
*/
}


void Network::createUniform(int depth, int inputSize, int nbOfFeatures)
{
	if (depth == 0) {
		//Case of 2 layer network, no hidden layers
		//Input Layer
		m_layers[0] = Layer(inputSize, nbOfFeatures);
	}
	else 
	{
		//Case of hidden layer network
		//From input layer to final hidden layer
		for (int i = 0; i < depth; i++) {
			//Create hidden layers with input and output of the same size (For MNIST, 28*28 in,28*28 out)
			m_layers[i] = Layer(inputSize, inputSize);
		}
	}

	//Output Layer
	m_layers[depth + 1] = Layer(inputSize, nbOfFeatures);
}

/*
void Network::feedforward(const float *input)
{
	//Copy input array to layerInput array
	memcpy(m_inputLayer.layerInput, input, m_inputLayer.inputCount * sizeof(float));

	m_inputLayer.calculate();

	update(-1);

	if (m_hiddenLayers)
	{
		for (int i = 0; i < m_hiddenLayerCount; i++)
		{
			m_hiddenLayers[i]->calculate();
			update(i);
		}
	}

	m_outputLayer.calculate();
}
*/

/*
float Network::backpropagate(const float *desiredOutput, const float *input, float alpha, float momentum)
{
	float generalError = 0;
	float localError;
	float sum = 0, localSum = 0;
	float delta, udelta;
	float output;

	feedforward(input);

	int i, j, k;

	for (int i = 0; i < m_outputLayer.neuronCount; i++)
	{
		output = m_outputLayer.neurons[i]->output;
		localError = (desiredOutput[i] - output) * output * (1 - output);
		generalError += (desiredOutput[i] - output) * (desiredOutput[i] - output);

		for (j = 0; j < m_outputLayer.inputCount; j++)
		{
			delta = m_outputLayer.neurons[i]->deltaValues[j];

			udelta = alpha * localError * m_outputLayer.layerInput[j] + delta * momentum;

			m_outputLayer.neurons[i]->weights[j] += udelta;
			m_outputLayer.neurons[i]->deltaValues[j] = udelta;

			sum += m_outputLayer.neurons[i]->weights[j] * localError;
		}

		m_outputLayer.neurons[i]->weightGain += alpha * localError * m_outputLayer.neurons[i]->gain);

	}

	for (i = (m_hiddenLayerCount - 1); i >= 0; i--)
	{
		for (j = 0; j < m_hiddenLayers[i]->neuronCount; j++)
		{
			output = m_hiddenLayers[i]->neurons[j]->output;

			localError = output * (1 - output) * sum;

			for (k = 0; k < m_hiddenLayers[i]->inputCount; k++)
			{
				delta = m_hiddenLayers[i]->neurons[j]->deltaValues[k];
				udelta = alpha * localError * m_hiddenLayers[i]->layerInput[k];
				m_hiddenLayers[i]->neurons[j]->weights[k] += udelta;
				m_hiddenLayers[i]->neurons[j]->deltaValues[k] = udelta;
				localSum += m_hiddenLayers[i]->neurons[j]->weights[k] * localError;
			}

			m_hiddenLayers[i]->neurons[j]->weightGain += alpha * localError * m_hiddenLayers[i]->neurons[j]->gain;
		}

		sum = localSum;
		localSum = 0;
	}

	for (i = 0; i < m_inputLayer.neuronCount; i++)
	{
		output = m_inputLayer.neurons[i]->output;
		localError = output * (1 - output) * sum;

		for (j = 0; j < m_inputLayer.inputCount; j++)
		{
			delta = m_inputLayer.neurons[i]->deltaValues[j];
			udelta = alpha * localError * m_inputLayer.layerInput[j] + delta * momentum;

			m_inputLayer.neurons[i]->weights[j] += udelta;
			m_inputLayer.neurons[i]->deltaValues[j] = udelta;
		}

		m_inputLayer.neurons[i]->weightGain += alpha * localError * m_inputLayer.neurons[i]->gain;
	}

	return generalError / 2;
}
*/