/**
 * @file	Network.h.
 *
 * @brief	Declares the structure of a network.
 */

#include "Neuron.h"
#include "Layer.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

/**
 * @class	Network
 *
 * @brief	A generic neural network, that may be trained with any data
 * 			that can be stored in a vector of doubles.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 */

class Network
{
private:
	vector<Layer> m_layers;
	vector<vector<double>> m_targetValues;
	double m_accuracy;
	int m_inputSize, m_outputSize, m_depth;

	double m_error;
	double m_recentAverageError;
	double m_recentAverageRate = 100; // Number of samples to average over

	bool m_testing = false;

	void initialiseWeights();
	void feedForward(vector<double> sample);
	void backPropagate(double expected);
	void updateWeights();
public:
	const double TARGET_ERROR = 0;
	const double TARGET_RECOGNITION = 10; // Target percentage recognition rate
	const int PRINT_RATE = 100; // Print recognition rate every PRINT_RATE samples
	const int MAX_EPOCHS = 100;
	const int MAX_VALIDATION_CHECKS = 6;
	const double MIN_CHANGE = 0.0001;

	Network();
	Network(unsigned depth, unsigned inputSize, unsigned nbOfFeatures);
	~Network();

	double getRecentAverageError() { return m_recentAverageError; }

	void createUniform(unsigned depth, unsigned inputSize, unsigned nbOfFeatures);

	void train(vector<vector<double>> data, vector<double> labels);

	void test(vector<vector<double>> data, vector<double> labels);

	void getResults(vector<double> &resultVals);

	double hardThreshold(double x);

	void save();
};

