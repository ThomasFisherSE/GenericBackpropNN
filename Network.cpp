#include "Network.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

using namespace std;

Network::Network()
{
	
}

Network::Network(int depth, int inputSize, int nbOfFeatures)
{
	m_depth = depth;
	m_inputSize = inputSize;
	m_outputSize = nbOfFeatures;
	createUniform(depth, inputSize, nbOfFeatures);
}

Network::~Network()
{
}

void Network::initialiseWeights()
{
	if (m_testing) { cout << "Initializing weights at random..." << endl; }
	
	for (int i = 0; i < m_layers.size(); i++) {
		m_layers[i].initialiseWeights();
	}
}

void Network::forwardPass(vector<double> sample)
{
	if (m_testing) { cout << "Forward propagating..." << endl; }

	// Initialise input layer
	m_layers[0].initialiseInputs(sample);

	// Propagate input to first hidden layer
	vector<double> prevX = m_layers[1].propagateWeigths(sample);

	// Propagate through all other layers
	for (int l = 2; l < m_layers.size(); l++) {
		prevX = m_layers[l].propagateWeigths(prevX);
	}
}

void Network::backwardPass(vector<double> sample, double expected)
{
	if (m_testing) { cout << "Backpropagating..." << endl; }

	// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L
	Layer finalLayer = m_layers[m_layers.size() - 1];
	finalLayer.calcFinalDelta(expected, m_layers[m_layers.size() - 2]);

	// For previous layers: (δ_i)^(l-1) = (1 - (((x_i)^(l-1))^2)(sum of{((w_ij)^l)((δ_j)^l)}
	for (int l = m_layers.size() - 2; l >= 0; l--) {
		if (m_testing) { cout << "for layer: " << l << endl; }
		m_layers[l].backPropagate(sample, m_layers[l]);
	}
}

void Network::updateWeights() {
	if (m_testing) { cout << "Updating weights..." << endl; }

	for (int l = 1; l < m_layers.size(); l++) {
		m_layers[l].updateWeights(m_layers[l - 1]);
	}
}

void Network::test(vector<vector<double>> data, vector<double> labels) {
	int numberCorrect = 0;

	//cout << "Target recognition rate: " << TARGET_RECOGNITION << endl;

	for (int i = 0; i < data.size(); i++) {
		vector<double> sample = data[i]; // Select the sample at this random index
		double expected = labels[i];
		double actual = -1;

		// Check actual output compared to expected output
		forwardPass(data[i]);

		Layer finalLayer = m_layers[m_layers.size()-1];

		actual = finalLayer.getX(0);

		//ACTUAL ALWAYS 0

		
		/*
		NOTE: NEED TO ADD FUNCTIONALITY FOR MULTIPLE OUTPUT NEURONS,
		BUT TRYING WITH 1 OUTPUT FOR NOW

		for (int i = 0; i < m_outputSize; i++) {
			double x = m_layers[m_layers.size() - 1].getX(i);

			if (x==1) {
				actual = i;
			}
		}*/

		if (actual >= 0) {
			actual = 1;
		}
		else {
			actual = 0;
		}

		if (true) { cout << "\nExpected: " << expected << " | Actual: " << actual << endl; }

		if (actual == expected) {
			numberCorrect++;
		}

		cout << "Testing sample: " << i+1 << " / " << data.size() << '\r';
	}
	
	cout << endl;

	m_recognitionRate = ((double)numberCorrect / data.size() * 100.0);
	cout << "Recognition Rate: " << m_recognitionRate << endl;
}

void Network::train(vector<vector<double>> data, vector<double> labels) {
	int numberCorrect = 0;
	int count = 0;

	/* 1: Initialize all weights (w_ij)^l at random */
	initialiseWeights();

	cout << "Target recognition rate: " << TARGET_RECOGNITION << endl;

	/* 2 : for t = 0, 1, 2, . . . do */
	//while (m_recognitionRate < TARGET_RECOGNITION) {
	while (count < 1000) {
		/* 3 : Pick n ∈{ 1, 2, · · · , N } */
		// i.e. pick a random sample
		int n = rand() % data.size(); // Generate a random number between 0 and the size of the data

		vector<double> sample = data[n]; // Select the sample at this random index
		double expected = labels[n];
		
		if (m_testing) { cout << "Expected output is " << expected << endl; }
		
		/* 4 :	Forward : Compute all (x_j)^l */
		forwardPass(sample);

		/* 5 :	Backward : Compute all (δ_j)^l */
		backwardPass(sample, expected);

		/* 6 :	Update the weights : (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l */
		updateWeights();

		// Check actual output compared to expected output
		double actual = m_layers[m_layers.size() - 1].getX(0);

		if (m_testing) { cout << "Expected: " << expected << "| Actual: " << actual << endl; }

		if (actual == expected) { 
			numberCorrect++;
		}

		count++;

		// Print recognition rate every few iterations
		if (count % PRINT_RATE == 0) {
			m_recognitionRate = ((double) numberCorrect / data.size()) * 100.0;
			//cout << "Current Recognition Rate: " << m_recognitionRate << '\r';
			int epoch = (int)(count / labels.size());
			cout << "Epoch: " << epoch << " | Completed training steps: " << count << " | Recognition Rate: " << m_recognitionRate << '\r';
		}

		/* 7:	Iterate to the next step until it is time to stop */
	}
	/* 8 : Return the final weights (w_ij)^l */#
	cout << endl;
}

void Network::createUniform(int depth, int inputSize, int nbOfFeatures)
{
	//m_layers.resize(depth); // e.g. for depth of 3, size = 2, m_layers = [L,L,L]

	//From input layer to final hidden layer
	for (int i = 0; i < depth-1; i++) {
		//Create hidden layers with input and output of the same size (For MNIST, 28*28 in,28*28 out)
		m_layers.push_back(Layer(inputSize, inputSize));
	}

	//Output Layer
	m_layers.push_back(Layer(inputSize, nbOfFeatures));
}