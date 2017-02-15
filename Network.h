#include "Neuron.h"
#include "Layer.h"

class Network
{
private:
	Layer m_inputLayer;
	Layer m_outputLayer;
	Layer **m_hiddenLayers;
	int m_hiddenLayerCount;
public:
	Network();
	~Network();

	void create(int inputCount, int inputNeurons, int outputCount, int *hiddenLayers, int hiddenLayerCount);
	void feedforward(const float *input);
	float backpropagate(const float *desiredOutput, const float *input, float alpha, float momentum);
	void update(int layerIndex);

	inline Layer &getOutput() 
	{
		return m_outputLayer;
	}

};

