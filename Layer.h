#include "Neuron.h"
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

class Layer
{
public:
	void initialiseWeights(); //Init weights to random numbers
	double sigmoid(double x) { return tanh(x); }
	double sigmoidDerivative(double x) { return 1.0 - x * x; } // tanh derivative
	double altSigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
	vector<double> forwardPropagate(Layer &prevLayer);
	vector<double> backPropagate(vector<double> input, Layer &prevLayer);
	double calculateError(double expected);
	void updateWeights(Layer &prevLayer);
	size_t size() { return m_outputs.size(); }
	double getWeight(unsigned i, unsigned j) { return m_weights.at(i, j); }
	void setWeight(unsigned i, unsigned j, double newWeight) { m_weights.at(i, j) = newWeight; }
	double getDelta(unsigned i) { return m_delta[i]; }
	double getGradient(unsigned i) { return m_gradients[i]; }
	void setDelta(unsigned i, double delta) { m_delta[i] = delta; }
	unsigned getOutputSize() { return m_outputSize; }
	double getOutput(unsigned i) { return m_outputs[i]; }
	vector<double> getOutputs() { return m_outputs; }
	void initialiseInputs(vector<double> sample) { m_outputs = sample; }

	void calcOutputGradients(double target);
	void calcHiddenGradients(Layer &nextLayer);
	double sumDerivativeOfWeights(Layer &nextLayer);
	Layer(unsigned inputSize, unsigned outputSize);
	Layer();
	~Layer();

	const double ETA = 0.1; // Learning rate constant

private:
	vector<double> m_outputs; // Neuron values
	vector<double> m_rawOutputs; // Before thresholding
	vector<double> m_delta;
	
	vector<double> m_gradients;
	unsigned m_inputSize, m_outputSize;
	mat m_weights;

};

