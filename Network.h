#include "Neuron.h"
#include "Layer.h"
#include <vector>
class Network
{
private:
	vector<Layer> m_layers;
	double m_error;
	vector<double> m_targetValues;

	/*
	Layer m_inputLayer;
	Layer m_outputLayer;
	Layer **m_hiddenLayers;
	int m_hiddenLayerCount;
	*/
public:
	Network();
	~Network();
	void createUniform(int depth, int inputSize, int nbOfFeatures);
	void feedforward(vector<vector<double>> data, vector<int> type, int &recognitionRate);

	void learn(vector<vector<double>> data);
};

