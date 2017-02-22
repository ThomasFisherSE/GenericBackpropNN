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
	if (m_hiddenLayers)
	{
		for (int i = 0; i < m_hiddenLayerCount; i++)
		{
			delete m_hiddenLayers[i];
		}

		delete[] m_hiddenLayers;
	}
}

void Network::create(int inputCount, int inputNeurons, int outputCount, int *hiddenLayers, int hiddenLayerCount)
{
	/*
	assert(inputCount && inputNeurons && outputCount);

	int i;

	m_inputLayer.create(inputCount, inputNeurons);

	if (hiddenLayers && hiddenLayerCount)
	{
		m_hiddenLayers = new Layer*[hiddenLayerCount];
		m_hiddenLayerCount = hiddenLayerCount;

		for (i = 0; i < hiddenLayerCount; i++)
		{
			m_hiddenLayers[i] = new Layer;

			if (i == 0)
			{
				m_hiddenLayers[i]->create(inputNeurons, hiddenLayers[i]);
			}
			else
			{
				m_hiddenLayers[i]->create(hiddenLayers[i - 1], hiddenLayers[i]);
			}
		}

		m_outputLayer.create(hiddenLayers[hiddenLayerCount - 1], outputCount);
	}
	else
	{
		m_outputLayer.create(inputNeurons, outputCount);
	}
	*/
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