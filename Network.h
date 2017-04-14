#include "Neuron.h"
#include "Layer.h"
#include <vector>
class Network
{
private:
	vector<Layer> m_layers;
	vector<vector<double>> m_targetValues;
	double m_recognitionRate;
	int m_inputSize, m_outputSize, m_depth;

	double m_error;
	double m_recentAverageError;
	double m_recentAverageRate = 100; // Number of samples to average over

	bool m_testing = false;

	void initialiseWeights();
	void forwardPass(vector<double> sample);
	void backwardPass(vector<double> sample, double expected);
	void updateWeights();
public:
	const double TARGET_ERROR = 0.5;
	const double TARGET_RECOGNITION = 10; // Target percentage recognition rate
	const int PRINT_RATE = 1; // Print recognition rate every PRINT_RATE samples

	Network();
	Network(unsigned depth, unsigned inputSize, unsigned nbOfFeatures);
	~Network();

	double getRecentAverageError() { return m_recentAverageError; }

	void createUniform(unsigned depth, unsigned inputSize, unsigned nbOfFeatures);

	void train(vector<vector<double>> data, vector<double> labels);

	void test(vector<vector<double>> data, vector<double> labels);

	void getResults(vector<double> &resultVals);
};

