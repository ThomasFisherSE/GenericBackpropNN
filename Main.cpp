#include "MNistReader.h"
#include "Network.h"
#include <iostream>

int main(int argc, char *argv[])
{
	const string DIVIDER = "*************************************************";
	cout << DIVIDER << endl << "MNIST Character Recognition Neural Network" << endl << DIVIDER << endl;

	vector<vector<double>> testData;
	vector<vector<double>> trainingData;
	vector<vector<double>> testLabels;
	vector<vector<double>> trainingLabels;

	MNistReader dataReader;

	cout << "Reading test data..." << endl;
	dataReader.readMnist(MNistReader::TEST_SAMPLES, MNistReader::NO_OF_PX, testData);

	cout << "Creating network..." << endl;
	Network net(3, MNistReader::NO_OF_PX, 1);
	cout << "Created successfully." << endl;

	cout << "Training network..." << endl;
	net.train(testData);
	cout << "Training complete." << endl;
}