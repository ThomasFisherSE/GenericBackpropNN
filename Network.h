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

	/*
	Layer m_inputLayer;
	Layer m_outputLayer;
	Layer **m_hiddenLayers;
	int m_hiddenLayerCount;
	*/
public:
	const double TARGET_RECOGNITION = 50; // Target percentage recognition rate

	Network();
	Network(int depth, int inputSize, int nbOfFeatures);
	~Network();
	void createUniform(int depth, int inputSize, int nbOfFeatures);

	void learn(vector<vector<double>> data);
};

