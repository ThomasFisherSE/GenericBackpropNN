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
	createUniform(depth, inputSize, nbOfFeatures);
}

Network::~Network()
{
}

void Network::train(vector<vector<double>> data, vector<double> labels) {
	int numberCorrect = 0;
	int count = 0;

	if (m_testing) { cout << "Initializing weights at random..." << endl; }
	/* 1: Initialize all weights (w_ij)^l at random */
	for (int i = 0; i < m_layers.size(); i++) {
		m_layers[i].initialiseLayer();
	}

	cout << "Target recognition rate: " << TARGET_RECOGNITION << endl;
	/* 2 : for t = 0, 1, 2, . . . do */
	while (m_recognitionRate < TARGET_RECOGNITION) {
		/* 3 : Pick n ∈{ 1, 2, · · · , N } */
		// i.e. pick a random sample
		int n = rand() % data.size(); // Generate a random number between 0 and the size of the data

		vector<double> sample = data[n]; // Select the sample at this random index

		// Initialise input layer
		m_layers[0].setX(sample);

		if (m_testing) { cout << "Forward propagating..." << endl; }
		
		/* 4 :	Forward : Compute all (x_j)^l */
		for (int l = 0; l < m_layers.size()-1; l++) {	
			m_layers[l+1].setX(m_layers[l].propagateWeigths(sample));
		}

		if (m_testing) { cout << "Backpropagating..." << endl; }

		/* 5 :	Backward : Compute all (δ_j)^l */
		// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L
		m_layers[m_layers.size()-1].setFinalLayer(labels[n], m_layers[m_layers.size() - 2]);

		// For previous layers: (δ_i)^(l-1) = (1 - (((x_i)^(l-1))^2)(sum of{((w_ij)^l)((δ_j)^l)}
		for (int l = m_layers.size() - 1; l > 0; l--) {
			if (m_testing) { cout << "for layer: " << l << endl; }
			m_layers[l].backPropagate(sample, m_layers[l-1]);
		}

		if (m_testing) { cout << "Updating weights..." << endl; }

		/* 6 :	Update the weights : (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l */
		for (int l = 1; l < m_layers.size(); l++) {
			m_layers[l].updateWeights(m_layers[l-1]);
		}

		// Print recognition rate every few iterations
		m_recognitionRate = (numberCorrect / data.size()) * 100;
		
		// Print recognition rate every x samples
		if (count % PRINT_RATE == 0) {
			cout << "Current Recognition Rate: " << m_recognitionRate << '\r';
		}

		count++;

		/* 7:	Iterate to the next step until it is time to stop */
	}

	/* 8 : Return the final weights (w_ij)^l */

}


void Network::createUniform(int depth, int inputSize, int nbOfFeatures)
{
	m_layers.resize(depth); // e.g. for depth of 3, size = 2, m_layers = [L,L,L]

	//From input layer to final hidden layer
	for (int i = 0; i < depth; i++) {
		//Create hidden layers with input and output of the same size (For MNIST, 28*28 in,28*28 out)
		m_layers[i] = Layer(inputSize, inputSize);
	}

	//Output Layer
	m_layers[depth-1] = Layer(inputSize, nbOfFeatures);
}