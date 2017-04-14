#include "Layer.h"
#include <math.h>
#include <assert.h>
#include <iostream>

using namespace std;

Layer::Layer(int inputSize, int outputSize)
{
	m_inputSize = inputSize;
	m_outputSize = outputSize;
	m_outputs.resize(outputSize);
	m_rawOutputs.resize(outputSize);
	m_delta.resize(outputSize);
	m_gradients.resize(outputSize);
}

Layer::Layer()
{
}


Layer::~Layer()
{

}

void Layer::initialiseWeights()
{
	m_weights.randn(m_inputSize, m_outputSize);
}

void Layer::updateWeights(Layer prevLayer)
{
	for (int i = 0; i < prevLayer.getOutputSize(); i++) {
		for (int j = 0; j < m_outputSize; j++) {
			// (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l
			// New weight = old weight - (eta * x from previous layer * delta for current layer)
			double oldWeight = prevLayer.getWeight(i, j);

			double newWeight = oldWeight - ETA * prevLayer.getOutput(i) * m_gradients[j];
		}
	}

	/*
	for (int row = 0; row < m_weights.n_rows; row++) {
		for (int col = 0; col < m_weights.n_cols; col++) {
			
			m_weights(row,col) -= (ETA * prevLayer.getOutput(row) * m_delta[col]);
			//cout << m_weights(row, col);
		}
	}
	*/
}



vector<double> Layer::propagateWeigths(vector<double> prevXvals)
{
	// (x_j)^l = θ(sum of{(w_ij)^l)((x_i)^(l-1))})
	//Next x = ThresholdOf(Sum of(Current weight * x from the previous layer))

	// For each neuron
	for (int j = 0; j < m_weights.n_cols; j++) 
	{
		double sum = 0.0;
		for (int i = 0; i < prevXvals.size(); i++)
		{
			double currentWeight = m_weights.at(i, j);
			sum += (currentWeight * prevXvals[i]);
		}

		sum += 1.0;//+1 for bias node

		m_rawOutputs[j] = sum;
		m_outputs[j] = sigmoid(sum);
	}

	return m_outputs;
}

double Layer::calcFinalDelta(double target)
{
	// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L
	// delta = 2(1-tanh^2((s_1)^2))(x_1 - y_n)

	double error = 0;

	for (int i = 0; i < m_outputSize; i++) {
		double delta = target - m_outputs[i];
		m_delta[i] = delta;
		error += delta * delta;
	}
	
	return error;

	/*
	double tanhS = sigmoid(m_rawOutputs[0] * m_rawOutputs[0]);

	m_delta[0] = 2 * (1 - tanhS*tanhS) * (m_outputs[0] - target);
	*/
}

double Layer::sumDerivativeOfWeights(Layer &nextLayer) {
	double sum = 0.0;

	for (int i = 0; i < m_outputSize - 1; i++) {
		for (int j = 0; j < nextLayer.getOutputSize() - 1; j++) {
			sum += m_weights.at(i, j) * nextLayer.getGradient(j);
		}
	}

	return sum;
}

void Layer::calcHiddenGradients(Layer &nextLayer) {
	double dow = sumDerivativeOfWeights(nextLayer);
	for (int i = 0; i < m_outputSize; i++) {
		m_gradients[i] = dow * sigmoidDerivative(m_outputs[i]);
	}	
}

void Layer::calcOutputGradients(double target) {
	for (int i = 0; i < m_outputSize; i++) {
		double delta = target - m_outputs[i];
	}	
}

vector<double> Layer::backPropagate(vector<double> input, Layer &prevLayer)
{
	// (δ_i)^(l-1) = (1 - (((x_i)^(l-1))^2)(sum of{((w_ij)^l)((δ_j)^l)}
	// Delta for prev layer = (1 - (x from prev layer squared * (sum of (weights from current layer * current deltas))))
	
	double sum;
	vector<double> output;
	output.resize(m_inputSize);

	for (int i = 0; i < m_weights.n_cols; i++) {
		sum = 0;

		for (int j = 0; j < m_weights.n_rows; j++) {
			sum += (m_weights(j, i) * m_delta[j]);
		}

		double prevX = prevLayer.getOutput(i);

		prevLayer.setDelta(i, 1 - prevX * prevX * sum);
	}

	return output;
}