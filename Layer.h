#include "Neuron.h"
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

class Layer
{
public:
	void initialiseLayer(); //Init weights to random numbers

	double sigmoid(double x);
	vector<double> &propagateWeigths(vector<vector<double>> input);
	vector<double> &backPropagate(vector<vector<double>> input);
	void updateWeights();
	int size() { return m_outputSize; }
	double getWeight(int x, int y) { return m_weights(x, y); }
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
	double m_prevX;
	double m_prevDelta;
	double m_delta;
	int m_inputSize, m_outputSize;
	mat m_weights;
	vector<double> m_thresholds;  //Possibly just '1'

};

