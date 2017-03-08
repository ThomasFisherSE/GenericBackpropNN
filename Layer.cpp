#include "Layer.h"
#include <math.h>
#include <assert.h>


Layer::Layer(int inputSize, int outputSize)
{
	m_inputSize = inputSize;
	m_outputSize = outputSize;
}

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

	int i, j;
	double sum = 0;

	for (i = 0; i < m_outputSize; i++)
	{
		for (j = 0; j < m_inputSize; j++)
		{
			sum += m_weights(i,j) * m_weights(j);
		}

		sum = sigmoid(sum);
		
	}

	
}

vector<double>& Layer::backPropagate(vector<vector<double>> input, double prevDelta)
{
	// (δ_i)^(l-1) = (1 - (((x_i)^(l-1))^2)(sum of{((w_ij)^l)((δ_j)^l)}
	// Delta for previous layer = (1 - (x from previous layer squared * (sum of (weights from current layer * next delta))))
	
	
}

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