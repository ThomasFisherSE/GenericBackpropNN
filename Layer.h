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
	double sigmoidDerivative(double x) { return 1.0 - x * x; }
	double altSigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
	vector<double> propagateWeigths(vector<double> prevXvals);
	vector<double> backPropagate(vector<double> input, Layer &prevLayer);
	void calcFinalDelta(double expected);
	void updateWeights(Layer prevLayer);
	int size() { return m_outputs.size(); }
	double getWeight(int x, int y) { return m_weights(x, y); }
	double getDelta(int i) { return m_delta[i]; }
	double getGradient(int i) { return m_gradients[i]; }
	void setDelta(int i, double delta) { m_delta[i] = delta; }
	int getOutputSize() { return m_outputSize; }
	double getOutput(int i) { return m_outputs[i]; }
	vector<double> getOutputs() { return m_outputs; }
	void initialiseInputs(vector<double> sample) { m_outputs = sample; }

	void calcOutputGradients(double target);
	void calcHiddenGradients(Layer &nextLayer);
	double sumDerivativeOfWeights(Layer &nextLayer);
	Layer(int inputSize, int outputSize);
	Layer();
	~Layer();

	const double ETA = 0.1; // Learning rate constant

private:
	vector<double> m_outputs; // Neuron values
	vector<double> m_rawOutputs; // Before thresholding
	vector<double> m_delta;
	double m_error;
	vector<double> m_gradients;
	double m_recentAverageError;
	double m_recentAverageRate = 100; // Number of samples to average over
	int m_inputSize, m_outputSize;
	mat m_weights;

};

