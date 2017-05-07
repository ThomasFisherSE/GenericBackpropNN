/**
 * @file	Layer.h.
 *
 * @brief	Declares the structure for a layer
 */

#include <vector>
#include <armadillo>

using namespace std;

using namespace arma;

/**
 * @class	Layer
 *
 * @brief	A layer of a neural network. 
 * 			Consists of neurons and weights (connections with other layers).
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 */

class Layer
{
public:
	Layer(unsigned inputSize, unsigned outputSize);
	Layer();
	~Layer();

	void initialiseWeights(); //Init weights to random numbers
	double sigmoid(double x) { return tanh(x); }
	double sigmoidDerivative(double x) { return 1.0 - x * x; } // tanh derivative
	double altSigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
	vector<double> feedForward(Layer &prevLayer);
	double calculateError(double expected);
	void updateWeights(Layer &prevLayer);
	size_t size() { return m_outputs.size(); }
	double getWeight(unsigned row, unsigned col) { return m_weights.at(row, col); }
	void setWeight(unsigned row, unsigned col, double newWeight) { m_weights.at(row, col) = newWeight; }
	double getWeightChange(unsigned row, unsigned col) { return m_weightChanges.at(row, col); }
	void setWeightChange(unsigned row, unsigned col, double newWeightChange) { m_weightChanges.at(row, col) = newWeightChange; }
	double getGradient(unsigned i) { return m_gradients[i]; }
	unsigned getOutputSize() { return m_outputSize; }
	double getOutput(unsigned i) { return m_outputs[i]; }
	vector<double> getOutputs() { return m_outputs; }
	void initialiseInputs(vector<double> sample) { m_outputs = sample; }
	void calcOutputGradient(double target);
	void backPropagate(Layer &nextLayer);
	double sumDerivativeOfWeights(Layer &nextLayer);
	void setEta(double eta) { m_eta = eta; }
private:
	vector<double> m_outputs; // Neuron values
	vector<double> m_rawOutputs; // Before thresholding
	vector<double> m_gradients;
	unsigned m_inputSize, m_outputSize;
	mat m_weights;
	mat m_weightChanges;
	double m_eta = 0.01;

	//const double ETA = 0.01; // Learning rate constant
	const double ALPHA = 0.5; // Momentum, fraction of last weight
};

