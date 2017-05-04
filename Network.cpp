/**
 * @file	Network.cpp.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 * 
 * @brief	Implements a generic neural network.
 */

#include "Network.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

using namespace std;

/**
 * @brief	Default constructor.
 */

Network::Network()
{
	
}

/**
 * @brief	Constructor to create a uniform network.
 * 			
 * @param	depth			The number of layers in the network.
 * @param	inputSize   	Number of neurons in the input layer.
 * @param	nbOfFeatures	The number of neurons in the output layer.
 */

Network::Network(unsigned depth, unsigned inputSize, unsigned nbOfFeatures)
{
	m_error = 0;
	m_recentAverageError = 0;

	m_depth = depth;
	m_inputSize = inputSize;
	m_outputSize = nbOfFeatures;
	createUniform(depth, inputSize, nbOfFeatures);
}

/**
 * @brief	Destructor.
 */

Network::~Network()
{
}

/**
 * @brief	Initialises the weights of the network.
 */

void Network::initialiseWeights()
{
	if (m_testing) { cout << "Initializing weights at random..." << endl; }
	
	for (unsigned i = 0; i < m_layers.size(); i++) {
		m_layers[i].initialiseWeights();
	}
}

/**
 * @brief	Feed-forward a sample through the network
 *
 * @param	sample	The sample to be fed through the network.
 */

void Network::feedForward(vector<double> sample)
{
	if (m_testing) { cout << "Forward pass..." << endl; }

	// Initialise input layer
	m_layers[0].initialiseInputs(sample);

	for (unsigned l = 1; l < m_layers.size(); l++) {
		Layer &prevLayer = m_layers[l - 1];

		m_layers[l].feedForward(prevLayer);
	}
}

/**
 * @brief	Back propagate errors through the network.
 * 			
 * @param	target	The expected output of the network.
 */

void Network::backPropagate(double target)
{
	if (m_testing) { cout << "Backwards Pass..." << endl; }

	Layer &outputLayer = m_layers.back();
	m_error = outputLayer.calculateError(target);

	m_recentAverageError = (m_recentAverageError * m_recentAverageRate + m_error) / (m_recentAverageRate + 1.0);

	// Calculate output gradient(s)
	outputLayer.calcFinalDelta(target);

	// Calculate hidden layer gradients
	for (size_t l = m_layers.size() - 2; l > 0; l--) {
		Layer &hiddenLayer = m_layers[l];
		Layer &nextLayer = m_layers[l + 1];

		hiddenLayer.backPropagate(nextLayer);
	}
}

/**
 * @brief	Updates the weights within the network.
 */

void Network::updateWeights() {
	if (m_testing) { cout << "Updating weights..." << endl; }

	for (size_t l = m_layers.size() - 1; l > 0; l--) {
		Layer &prevLayer = m_layers[l - 1];
		m_layers[l].updateWeights(prevLayer);
	}
}

/**
 * @brief	Gets the output of the network and put into a vector
 * 			
 * @param [in,out]	resultVals	Vector to hold result values in
 */

void Network::getResults(vector<double> &resultVals) {
	resultVals.clear();

	for (unsigned i = 0; i < m_layers.back().getOutputSize(); i++) {
		resultVals.push_back(m_layers.back().getOutput(i));
	}
}

/**
 * @brief	Test the network with some test data.
 *
 * @param	data  	The data to test the network with.
 * @param	labels	The labels assosciated with the test data.
 */

void Network::test(vector<vector<double>> data, vector<double> labels) {
	unsigned numberCorrect = 0;
	unsigned numberIncorrect = 0;

	for (unsigned i = 0; i < data.size(); i++) {
		vector<double> sample = data[i]; // Select the sample at this random index
		double target = labels[i];
		double output;

		// Check actual output compared to expected output
		feedForward(data[i]);

		vector<double> results; // Vector of results

		getResults(results); // Put results in results vector

		output = hardThreshold(results[0]);

		if (true) { cout << "\nTarget: " << target << " | Output: " << output << endl; }

		if (output == target) {
			numberCorrect++;
		}

		cout << "Testing sample: " << i+1 << " / " << data.size() << '\r';
	}
	
	cout << endl;

	m_accuracy = ((double)numberCorrect / data.size() * 100.0);
cout << "Accuracy: " << m_accuracy << endl;
}

/**
 * @brief	Train the network with some training data.
 *
 * @param	data  	The data to train the network with.
 * @param	labels	The labels assosciated with the training data.
 */

void Network::train(vector<vector<double>> data, vector<double> labels) {
	unsigned numberCorrect = 0;
	unsigned count = 0;
	double previousError = 999;
	double changeInError = 999;
	unsigned validationChecks = 0;

	ofstream out("errortracking.log");

	/* 1: Initialize all weights (w_ij)^l at random */
	initialiseWeights();

	/* 2 : for t = 0, 1, 2, . . . do */
	unsigned epoch = 0;

	while (validationChecks < MAX_VALIDATION_CHECKS
		&& (epoch < MAX_EPOCHS)) {
		/* 3 : Pick n ∈{ 1, 2, · · · , N } */
		// i.e. pick a random sample
		unsigned n = rand() % data.size(); // Generate a random number between 0 and the size of the data

		vector<double> sample = data[n]; // Select the sample at this random index
		double target = labels[n];

		/* 4 :	Forward : Compute all (x_j)^l */
		feedForward(sample);

		/* 5 :	Backward : Compute all (δ_j)^l */
		backPropagate(target);

		/* 6 :	Update the weights : (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l */
		updateWeights();

		changeInError = m_recentAverageError - previousError;

		if (changeInError < 0) {
			changeInError *= -1;
		}

		count++;

		// Increment validation checks
		if (changeInError < MIN_CHANGE) {
			validationChecks++;
		}
		else {
			validationChecks = 0;
		}

		// Set next previous error
		previousError = m_recentAverageError;

		// Print pass details
		if (count % PRINT_RATE == 0 || validationChecks > 4) {
			if (true) {
				double output = m_layers.back().getOutput(0);
				cout << "Expected: " << labels[n] << " | Obtained: " << output << endl;
			}
			//m_recognitionRate = ((double) numberCorrect / data.size()) * 100.0;
			//cout << "Current Recognition Rate: " << m_recognitionRate << '\r';
			epoch = (unsigned)(count / labels.size());
			cout << "Epoch: " << epoch << " | Completed training steps: " << count <<
				" | Recent Average Error: " << m_recentAverageError <<
				" | Validation Checks: " << validationChecks <<
				" | Change in Error: " << changeInError << endl;
		}

		out << m_recentAverageError << endl;

		/* 7:	Iterate to the next step until it is time to stop */
	}
	/* 8 : Return the final weights (w_ij)^l */
	cout << endl;

	out.close();
}

/**
 * @brief	Creates a uniform network.
 *
 * @param	depth			The number of layers in the network.
 * @param	inputSize   	Number of neurons in the input layer.
 * @param	nbOfFeatures	The number of neurons in the output layer.
 */

void Network::createUniform(unsigned depth, unsigned inputSize, unsigned nbOfFeatures)
{
	// Create input layer
	m_layers.push_back(Layer(inputSize, inputSize));

	//Create hidden layers
	for (unsigned l = 0; l < depth - 2; l++) {
		//Create hidden layers with same size input and output
		m_layers.push_back(Layer(inputSize, inputSize));
	}

	//Output Layer
	m_layers.push_back(Layer(inputSize, nbOfFeatures));
}

/**
 * @brief	Hard threshold a value x.
 *
 * @param	x	The value to threshold.
 *
 * @return	A double, either 0.0 or 1.0.
 */

double Network::hardThreshold(double x) {
	if (x >= 0.5) {
		return 1;
	}
	else {
		return 0;
	}
}

/**
 * @brief	Saves the weights of the network to file.
 */

void Network::save() {
	ofstream out("net.network");
	for (int l = 0; l < m_layers.size() - 1; l++) {
		out << "OMEGA" << l + 1 << endl;

		for (int i = 0; i < m_layers.at(l).getOutputSize(); i++) {
			for (int j = 0; j < m_layers.at(l + 1).getOutputSize(); j++) {
				out << std::fixed << std::setprecision(5) << m_layers.at(l).getWeight(i,j) << endl;
			}
		}
	}

	out.close();
}