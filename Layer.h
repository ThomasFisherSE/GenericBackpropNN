#include "Neuron.h"
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

class Layer
{
public:
	void initialiseLayer(); //Init weights to random numbers

	vector<double> &propagateWeigths(vector<double> input);
	vector<double> &backPropagate(vector<double> input);
	Layer(int m,int n);
	Layer();
	~Layer();

	/*
	void create(int inputSize, int numberOfNeurons);
	void calculate();
	*/
private:
	int m_inputSize, m_outputSize;
	mat m_weights;
	vector<double> m_thresholds;  //Possibly just '1'

};

