#include "Neuron.h"
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

class Layer
{
public:
	void initialiseLayer(); //Init weights to random numbers

	double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
	vector<double> &propagateWeigths(vector<double> input);
	vector<double> &backPropagate(vector<double> input, Layer prevLayer);
	void updateWeights();
	int size() { return m_outputSize; }
	double getWeight(int x, int y) { return m_weights(x, y); }
	double getDelta() { return m_delta; }
	void setDelta(double delta) { m_delta = delta; }
	double getX(int i) { return m_x.at(i); }
	void setX(int i, double x) { m_x.at(i) = x; }
	void setX(vector<double> xVector) { m_x = xVector; }
	Layer(int inputSize, int outputSize);
	Layer();
	~Layer();

	static const double eta; // Learning rate constant

	/*
	void create(int inputSize, int numberOfNeurons);
	void calculate();
	*/
private:
	double m_eta;
	vector<double> m_x;
	double m_prevX;
	double m_prevDelta;
	double m_delta;
	int m_inputSize, m_outputSize;
	mat m_weights;
	vector<double> m_thresholds;  //Possibly just '1'

};

