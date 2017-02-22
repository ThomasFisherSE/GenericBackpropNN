#include "Neuron.h"
#include "Layer.h"
#include <vector>
class Network
{
private:
	vector<Layer> layers;
	Layer m_inputLayer;
	Layer m_outputLayer;
	Layer **m_hiddenLayers;
	int m_hiddenLayerCount;
public:
	Network();
	~Network();
	void createUniform(int depth, int inputSize, int nbOfFeatures);
	void feedforward(vector<vector<double>> data, vector<int> type, int &recognitionRate);

	void create(int inputCount, int inputNeurons, int outputCount, int *hiddenLayers, int hiddenLayerCount);

	/*
	void feedforward(const float *input);
	float backpropagate(const float *desiredOutput, const float *input, float alpha, float momentum);
	void update(int layerIndex);
	*/
};

