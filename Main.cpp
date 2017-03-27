#include "MNistReader.h"
#include "Network.h"
#include <iostream>

int main(int argc, char *argv[])
{
	const string DIVIDER = "*************************************************";
	cout << DIVIDER << endl << "MNIST Character Recognition Neural Network" << endl << DIVIDER << endl;

	vector<vector<double>> testingImages;
	vector<vector<double>> trainingImages;
	vector<double> testLabels;
	vector<double> trainingLabels;

	MNistReader dataReader;

	cout << "Reading training data..." << endl;
	dataReader.readImages(MNistReader::TRAINING_SIZE, MNistReader::TOTAL_PIXELS, trainingImages, dataReader.TRAINING_IMAGES);

	cout << DIVIDER << endl;

	cout << "Reading training labels..." << endl;
	dataReader.readLabels(MNistReader::TRAINING_SIZE, trainingLabels, dataReader.TRAINING_LABELS);

	for (int i = 0; i < trainingLabels.size(); i++) {
		if (trainingLabels[i] != 0) {
			trainingLabels[i] = 1;
		}
	}

	cout << DIVIDER << endl;

	cout << "Reading test data..." << endl;
	dataReader.readImages(MNistReader::TESTING_SIZE, MNistReader::TOTAL_PIXELS, testingImages, dataReader.TEST_IMAGES);

	cout << DIVIDER << endl;

	cout << "Reading test labels..." << endl;
	dataReader.readLabels(MNistReader::TESTING_SIZE, testLabels, dataReader.TEST_LABELS);

	cout << DIVIDER << endl;

	cout << "Creating network..." << endl;
	Network net(3, MNistReader::TOTAL_PIXELS, 1);
	cout << "Created successfully." << endl;

	cout << DIVIDER << endl;
	cout << "Training network..." << endl;
	cout << DIVIDER << endl;

	net.train(trainingImages, trainingLabels);

	cout << DIVIDER << endl;
	cout << "Training complete." << endl;
	cout << DIVIDER << endl;
}