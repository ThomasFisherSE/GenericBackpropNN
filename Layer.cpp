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
	cout << m_x.size();
	m_delta.resize(inputSize);
}

Layer::Layer()
{
}


Layer::~Layer()
{

}

void Layer::initialiseLayer()
{
	m_weights.randn(m_inputSize, 1);
}

void Layer::updateWeights(Layer prevLayer)
{
	for (int col = 0; col < m_weights.n_cols; col++) {
		for (int row = 0; row < m_weights.n_rows; row++) {
			// (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l
			// New weight = old weight - (eta * x from previous layer * delta for current layer)
			m_weights[col, row] -= m_eta * prevLayer.getX(col) * m_delta[row];
		}
	}
}



vector<double>& Layer::propagateWeigths(vector<double> input)
{
	// (x_j)^l = θ(sum of{(w_ij)^l)((x_i)^(l-1))})
	//Next x = ThresholdOf(Sum of(Current weight * x from the previous layer))

	int i, j;
	vector<double> output;

	output.resize(m_inputSize);

	for (i = 0; i < m_weights.n_rows; i++)
	{
		double sum = 0;
		for (j = 0; j < m_weights.n_cols; j++)
		{
			sum += m_weights[i, j] * input[i] + 1;
		}
		
		output[i] = sigmoid(sum);
	}
	return output;

	
}

vector<double>& Layer::backPropagate(vector<double> input, Layer prevLayer)
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

		prevLayer.setDelta(i, 1 - pow(prevLayer.getX(i),2) * sum); //OUT OF BOUNDS AT getX(i), for some reason, i changes to a huge number.
	}

	return output;
}