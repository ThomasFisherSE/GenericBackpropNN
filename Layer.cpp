/**
 * @file	Layer.cpp.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 * 			
 * @brief	Implements a generic layer for a neural network, consisting of neurons and weights.
 */

#include "Layer.h"
#include <math.h>
#include <assert.h>
#include <iostream>

using namespace std;

/**
 * @fn	Layer::Layer(unsigned inputSize, unsigned outputSize)
 *
 * @brief	Constructor.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 *
 * @param	inputSize 	Size of the input.
 * @param	outputSize	Size of the output.
 */

Layer::Layer(unsigned inputSize, unsigned outputSize)
{
	m_inputSize = inputSize;
	m_outputSize = outputSize;
	m_outputs.resize(outputSize);
	m_rawOutputs.resize(outputSize);
	m_gradients.resize(outputSize);

	// Initialise outputs and gradients of each neuron to 0s
	for (unsigned n = 0; n < outputSize; n++) {
		m_outputs[n] = 0.0;
		m_rawOutputs[n] = 0.0;
		m_gradients[n] = 0.0;
	}
}

/**
 * @fn	Layer::Layer()
 *
 * @brief	Default constructor.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 */

Layer::Layer()
{
}

/**
 * @fn	Layer::~Layer()
 *
 * @brief	Destructor.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 */

Layer::~Layer()
{

}

/**
 * @fn	void Layer::initialiseWeights()
 *
 * @brief	Initialises the weights.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 */

void Layer::initialiseWeights()
{
	// Create matrix of weights with m_inputSize rows and m_outputSize columns
	m_weights.randn(m_inputSize, m_outputSize);
	m_weightChanges.zeros(m_inputSize, m_outputSize);
}

/**
 * @fn	void Layer::updateWeights(Layer &prevLayer)
 *
 * @brief	Updates the weights described by prevLayer.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 *
 * @param [in,out]	prevLayer	The previous layer.
 */

void Layer::updateWeights(Layer &prevLayer)
{
	for (unsigned j = 0; j < m_outputSize; j++) {
		for (unsigned i = 0; i < prevLayer.getOutputSize(); i++) {
			// (w_ij)^l <- (w_ij)^l - eta((x_i)^(l-1)) (delta_j)^l
			// New weight = old weight - (eta * x from previous layer * delta for current layer)		
			double oldWeight = prevLayer.getWeight(i, j);
			double oldWeightChange = prevLayer.getWeightChange(i, j);

			double newWeightChange = ETA * prevLayer.getOutput(i) * m_gradients[j] + ALPHA * oldWeightChange;

			//double newWeight = oldWeight - ETA * prevLayer.getOutput(i) * m_gradients[j];

			double newWeight = oldWeight + newWeightChange;

			prevLayer.setWeightChange(i, j, newWeightChange);

			prevLayer.setWeight(i, j, newWeight);
		}
	}
}

/**
 * @fn	vector<double> Layer::feedForward(Layer &prevLayer)
 *
 * @brief	Feed forward.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 *
 * @param [in,out]	prevLayer	The previous layer.
 *
 * @return	A vector&lt;double&gt;
 */

vector<double> Layer::feedForward(Layer &prevLayer) {
	// (x_j)^l = θ(sum of{(w_ij)^l * ((x_i)^(l-1))})
	//Next x = ThresholdOf(Sum of(Current weight * x from the previous layer))

	// For each neuron in this layer
	for (unsigned j = 0; j < m_outputSize; j++)
	{
		double sum = 0.0;

		// For each neuron in the previous layer
		for (unsigned i = 0; i < prevLayer.getOutputSize(); i++)
		{
			// Add the previous layer's output * the weight between (x_i)^(l-1) and (x_j)^l
			sum += prevLayer.getOutput(i) * prevLayer.getWeight(i, j);
		}

		sum += 1.0; // +1 for bias neuron

		m_rawOutputs[j] = sum;
		m_outputs[j] = sigmoid(sum);
	}

	return m_outputs;
}

/**
 * @fn	double Layer::calculateError(double target)
 *
 * @brief	Calculates the error.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 *
 * @param	target	Target for the.
 *
 * @return	The calculated error.
 */

double Layer::calculateError(double target)
{
	// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L
	double error = 0.0;

	for (unsigned n = 0; n < m_outputSize; n++) {
		double delta = target - m_outputs[n];
		error += delta * delta;
	}
	
	error /= m_outputSize; // Average error squared
	error = sqrt(error); // RMS

	return error;
}

/**
 * @fn	double Layer::sumDerivativeOfWeights(Layer &nextLayer)
 *
 * @brief	Sum derivative of weights.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 *
 * @param [in,out]	nextLayer	The next layer.
 *
 * @return	The total number of derivative of weights.
 */

double Layer::sumDerivativeOfWeights(Layer &nextLayer) {
	double sum = 0.0;

	// For each neuron in the current layer
	for (unsigned i = 0; i < m_outputSize; i++) {
		// For each neuron in the next layer
		for (unsigned j = 0; j < nextLayer.getOutputSize(); j++) {
			// Add the weight w_ij in the current layer * the gradient of 
			// x_j in the next layer to the sum
			sum += m_weights.at(i, j) * nextLayer.getGradient(j);
		}
	}

	return sum;
}

/**
 * @fn	void Layer::backPropagate(Layer &nextLayer)
 *
 * @brief	Back propagate.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 *
 * @param [in,out]	nextLayer	The next layer.
 */

void Layer::backPropagate(Layer &nextLayer) {
	double dow = sumDerivativeOfWeights(nextLayer);

	for (unsigned n = 0; n < m_outputSize; n++) {
		m_gradients[n] = dow * sigmoidDerivative(m_outputs[n]);
	}
}

/**
 * @fn	void Layer::calcFinalDelta(double target)
 *
 * @brief	Calculates the final delta.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 *
 * @param	target	Target for the.
 */

void Layer::calcFinalDelta(double target) {
	for (unsigned n = 0; n < m_outputSize; n++) {
		double delta = target - m_outputs[n];
		m_gradients[n] = delta * sigmoidDerivative(m_outputs[n]);
	}	
}