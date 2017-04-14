#include "MNistReader.h"
#include "Network.h"
#include <iostream>

const string DIVIDER = "*************************************************";

void nandGates() {
	vector<vector<double>> nandData(1000);
	vector<double> nandLabels(1000);

	for (int i = 0; i < 1000; i++) {
		nandData[i].resize(2);
	}

	for (int j = 0; j < 200; j++) {
		nandData[j][0] = 0;
		nandData[j][1] = 0;
		nandLabels[j] = 1;

		int k = j + 200;

		nandData[k][0] = 0;
		nandData[k][1] = 1;
		nandLabels[k] = 1;

		int l = k + 200;

		nandData[l][0] = 1;
		nandData[l][1] = 0;
		nandLabels[l] = 1;

		int m = l + 200;

		nandData[m][0] = 1;
		nandData[m][1] = 1;
		nandLabels[m] = 0;
	}


	vector<vector<double>> nandTestData(1000);
	vector<double> nandTestLabels(1000);

	for (int i = 0; i < 1000; i++) {
		nandTestData[i].resize(2);
	}

	for (int j = 0; j < 200; j++) {
		nandTestData[j][0] = 1;
		nandTestData[j][1] = 1;
		nandTestLabels[j] = 0;

		int k = j + 200;

		nandTestData[k][0] = 0;
		nandTestData[k][1] = 0;
		nandTestLabels[k] = 1;

		int l = k + 200;

		nandTestData[l][0] = 0;
		nandTestData[l][1] = 1;
		nandTestLabels[l] = 1;

		int m = l + 200;

		nandTestData[m][0] = 1;
		nandTestData[m][1] = 0;
		nandTestLabels[m] = 1;
	}

	Network net(3, 2, 1);

	net.train(nandData, nandLabels);
	net.test(nandTestData, nandTestLabels);
}

void characterRecognition() {
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

	for (int i = 0; i < testLabels.size(); i++) {
		if (testLabels[i] != 0) {
			testLabels[i] = 1;
		}
	}

	cout << DIVIDER << endl;

	cout << "Creating network..." << endl;
	Network net(3, MNistReader::TOTAL_PIXELS, 1); // 3 Layers, 512 Input Nodes, 1 Output Node
	cout << "Created successfully." << endl;

	cout << DIVIDER << endl;
	cout << "Training network..." << endl;
	cout << DIVIDER << endl;

	net.train(trainingImages, trainingLabels);

	cout << DIVIDER << endl;
	cout << "Training complete." << endl;
	cout << DIVIDER << endl;

	cout << "Testing Network:" << endl;
	cout << DIVIDER << endl;

	net.test(testingImages, testLabels);
}

int main(int argc, char *argv[])
{
	cout << DIVIDER << endl << "MNIST Character Recognition Neural Network" << endl << DIVIDER << endl;

	nandGates();
	//characterRecognition();

	string in;
	cin >> in;
}