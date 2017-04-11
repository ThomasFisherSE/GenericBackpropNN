#include "Neuron.h"
#include "Layer.h"
#include <vector>
class Network
{
private:
	vector<Layer> m_layers;
	double m_error;
	vector<vector<double>> m_targetValues;
	double m_recognitionRate;
	int m_inputSize, m_outputSize, m_depth;

	bool m_testing = false;

	void initialiseWeights();
	void forwardPass(vector<double> sample);
	void backwardPass(vector<double> sample, double expected);
	void updateWeights();
public:
	const double TARGET_ERROR = 10;
	const double TARGET_RECOGNITION = 10; // Target percentage recognition rate
	const int PRINT_RATE = 1; // Print recognition rate every PRINT_RATE samples

	Network();
	Network(int depth, int inputSize, int nbOfFeatures);
	~Network();
	void createUniform(int depth, int inputSize, int nbOfFeatures);

	void train(vector<vector<double>> data, vector<double> labels);

	void test(vector<vector<double>> data, vector<double> labels);
};

