#include "Layer.h"
#include <math.h>
#include <assert.h>
#include <iostream>

using namespace std;

Layer::Layer(int inputSize, int outputSize)
{
	m_inputSize = inputSize;
	m_outputSize = outputSize;
	m_x.resize(inputSize);
	m_s.resize(inputSize);
	m_delta.resize(outputSize);
}

Layer::Layer()
{
}


Layer::~Layer()
{

}

void Layer::initialiseLayer()
{
	m_weights.randn(m_inputSize, m_outputSize);
}

void Layer::updateWeights(Layer prevLayer)
{
	for (int col = 0; col < m_weights.n_cols; col++) {
		for (int row = 1; row < m_weights.n_rows; row++) {
			// (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l
			// New weight = old weight - (eta * x from previous layer * delta for current layer)
			m_weights[col, row] -= m_eta * prevLayer.getX(col) * m_delta[row-1];
		}
	}
}



vector<double> Layer::propagateWeigths(vector<double> input)
{
	// (x_j)^l = θ(sum of{(w_ij)^l)((x_i)^(l-1))})
	//Next x = ThresholdOf(Sum of(Current weight * x from the previous layer))

	int i, j;

	for (i = 0; i < m_weights.n_rows; i++)
	{
		double sum = 0;
		for (j = 0; j < m_weights.n_cols; j++)
		{
			sum += m_weights[i, j] * input[i] + 1;
		}
		
		m_s[i] = sum;
		m_x[i] = sigmoid(sum);
	}

	return m_x;
}

void Layer::setFinalLayer(double expected, Layer prevLayer)
{
	// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L
	// delta = 2(1-tanh^2((s_1)^2))(x_1 - y_n)

	m_delta[0] = 2 * (1 - sigmoid(sigmoid(prevLayer.getS(1) * prevLayer.getS(1)))) * (m_x[0] - expected);
}

vector<double> Layer::backPropagate(vector<double> input, Layer prevLayer)
{
	// (δ_i)^(l-1) = (1 - (((x_i)^(l-1))^2)(sum of{((w_ij)^l)((δ_j)^l)}
	// Delta for prev layer = (1 - (x from prev layer squared * (sum of (weights from current layer * current deltas))))
	
	double sum;
	vector<double> output;
	output.resize(m_inputSize);

	for (int i = 0; i < m_weights.n_rows; i++) {
		sum = 0;

		for (int j = 0; j < m_weights.n_cols; j++) {
			sum += (m_weights[i, j] * m_delta[j]);
		}

		double x = prevLayer.getX(i);

		prevLayer.setDelta(i, 1 - x * x * sum);
	}

	return output;
}