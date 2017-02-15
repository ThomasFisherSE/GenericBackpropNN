#include "Neuron.h"

class Layer
{
public:
	Layer();
	~Layer();
	Neuron **neurons;
	int neuronCount;
	float *layerInput;
	int inputCount;

	void create(int inputSize, int numberOfNeurons);
	void calculate();
	};

};

