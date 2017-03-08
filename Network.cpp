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
	/*
	if (m_hiddenLayers)
	{
		for (int i = 0; i < m_hiddenLayerCount; i++)
		{
			delete m_hiddenLayers[i];
		}

		delete[] m_hiddenLayers;
	}
	*/
}

void Network::learn(vector<vector<double>> data) {
	int numberCorrect = 0;

	/* 1: Initialize all weights (w_ij)^l at random */
	for (int i = 0; i < m_layers.size; i++) {
		m_layers.at(i).initialiseLayer();
	}

	/* 2 : for t = 0, 1, 2, . . . do */
	while (m_recognitionRate < TARGET_RECOGNITION) {
		for (int l = 0; l < m_layers.size; l++) {
			/* 3 : Pick n ∈{ 1, 2, · · · , N } */
			int n = rand();

			/* 4 :	Forward : Compute all (x_j)^l */
			m_layers.at(l).propagateWeigths(data);
		}

		/* 5 :	Backward : Compute all (δ_j)^l */
		// For final layer: (δ_1)^L = ∂e(w) / ∂(s_1)^L
	
		for (int l = m_layers.size-1; l >= 0; l++) {
			m_layers.at(l).backPropagate(data, m_layers.at(l+1).getDelta());
		}

		/* 6 :	Update the weights : (w_ij)^l ← (w_ij)^l - η ((x_i)^(l-1)) (δ_j)^l */
		for (int l = 0; l < m_layers.size; l++) {
			m_layers.at(l).updateWeights();
		}


		// Print recognition rate every few iterations
		m_recognitionRate = (numberCorrect / 60000) * 100;
		cout << "Recognition Rate: " << m_recognitionRate << endl;

		/* 7:	Iterate to the next step until it is time to stop */
	}

	/* 8 : Return the final weights (w_ij)^l */

}


void Network::createUniform(int depth, int inputSize, int nbOfFeatures)
{
	if (depth == 0) {
		//Case of 2 layer network, no hidden layers
		//Input Layer
		m_layers[0] = Layer(inputSize, nbOfFeatures);
	}
	else 
	{
		//Case of hidden layer network
		//From input layer to final hidden layer
		for (int i = 0; i < depth; i++) {
			//Create hidden layers with input and output of the same size (For MNIST, 28*28 in,28*28 out)
			m_layers[i] = Layer(inputSize, inputSize);
		}
	}

	//Output Layer
	m_layers[depth + 1] = Layer(inputSize, nbOfFeatures);
}