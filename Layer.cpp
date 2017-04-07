#include "Layer.h"
#include <math.h>
#include <assert.h>
#include <iostream>

using namespace std;

Layer::Layer(int inputSize, int outputSize)
{
	m_inputSize = inputSize;
	m_outputSize = outputSize;
	m_x.resize(outputSize);
	m_s.resize(outputSize);
	m_delta.resize(outputSize);
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
	for (int row = 0; row < m_weights.n_rows; row++) {
		for (int col = 0; col < m_weights.n_cols; col++) {
			// (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l
			// New weight = old weight - (eta * x from previous layer * delta for current layer)
			//double oldWeight = m_weights(row, col);
			m_weights(row,col) -= (ETA * prevLayer.getX(row) * m_delta[col]);
			//cout << m_weights(row, col);
		}
	}
}



vector<double> Layer::propagateWeigths(vector<double> prevXvals)
{
	// (x_j)^l = θ(sum of{(w_ij)^l)((x_i)^(l-1))})
	//Next x = ThresholdOf(Sum of(Current weight * x from the previous layer))

	for (int j = 0; j < m_weights.n_cols; j++) 
	{
		double sum = 0;
		for (int i = 0; i < m_weights.n_rows; i++)
		{
			double currentWeight = m_weights.at(i, j);
			sum += (currentWeight * prevXvals[i]);
		}

		sum += 1.0;//+1 for bias node

		m_s[j] = sum;
		m_x[j] = sigmoid(sum);
	}

	return m_x;
}

void Layer::calcFinalDelta(double expected, Layer prevLayer)
{
	// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L
	// delta = 2(1-tanh^2((s_1)^2))(x_1 - y_n)

	double tanhS = sigmoid(m_s[0] * m_s[0]);

	m_delta[0] = 2 * (1 - tanhS*tanhS) * (m_x[0] - expected);
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

		double prevX = prevLayer.getX(i);

		prevLayer.setDelta(i, 1 - prevX * prevX * sum);
	}

	return output;
}