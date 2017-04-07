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
	double altSigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
	vector<double> propagateWeigths(vector<double> prevXvals);
	vector<double> backPropagate(vector<double> input, Layer &prevLayer);
	void calcFinalDelta(double expected, Layer prevLayer);
	void updateWeights(Layer prevLayer);
	int size() { return m_x.size(); }
	double getWeight(int x, int y) { return m_weights(x, y); }
	double getDelta(int i) { return m_delta[i]; }
	void setDelta(int i, double delta) { m_delta[i] = delta; }
	int getOutputSize() { return m_outputSize; }
	double getX(int i) { return m_x[i]; }
	vector<double> getX() { return m_x; }
	double getS(int i) { return m_s[i]; }
	void setS(int i, double x) { m_s[i] = x; }
	void setX(int i, double x) { m_x[i] = x; }
	void initialiseInputs(vector<double> sample) { m_x = sample; }
	Layer(int inputSize, int outputSize);
	Layer();
	~Layer();

	const double ETA = 0.1; // Learning rate constant

private:
	vector<double> m_x; // Neuron values
	vector<double> m_s; // Before thresholding
	vector<double> m_delta;
	int m_inputSize, m_outputSize;
	mat m_weights;

};

