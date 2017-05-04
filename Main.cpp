/**
 * @file	Main.cpp.
 * @author	Thomas Fisher
 * @date	04/05/2017
 * @brief	Implements the main class. The entry-point of the application.
 */

#include "MNistReader.h"
#include "Network.h"
#include <iostream>

const string DIVIDER = "*************************************************";

/**
 * @brief	Assign NAND labels.
 *
 * @param			data  	The vector of data.
 * @param [in,out]	labels	The vector to put labels into.
 */

void assignNandLabels(vector<vector<double>> data, vector<double> &labels) {
	for (int i = 0; i < data.size(); i++) {
		if ((data[i][0] == 0) && data[i][1] == 0) {
			labels[i] = 1;
		}

		if ((data[i][0] == 0) && data[i][1] == 1) {
			labels[i] = 1;
		}

		if ((data[i][0] == 1) && data[i][1] == 0) {
			labels[i] = 1;
		}

		if ((data[i][0] == 1) && data[i][1] == 1) {
			labels[i] = 0;
		}
	}
}

/**
 * @brief	Generates NAND data.
 * 			
 * @param [in,out]	trnData  	Vector to put training data into.
 * @param [in,out]	trnLabels	Vector to put training labels into.
 * @param [in,out]	tstData  	Vector to put testing data into.
 * @param [in,out]	tstLabels	Vector to put testing labels into.
 */

void generateNandData(
	vector<vector<double>> &trnData, vector<double> &trnLabels, 
	vector<vector<double>> &tstData, vector<double> &tstLabels) {

	// Training Data:
	for (int i = 0; i < 1000; i++) {
		trnData[i].resize(2);
	}

	for (int j = 0; j < 200; j++) {
		trnData[j][0] = 0;
		trnData[j][1] = 0;

		int k = j + 200;

		trnData[k][0] = 0;
		trnData[k][1] = 1;

		int l = k + 200;

		trnData[l][0] = 1;
		trnData[l][1] = 0;

		int m = l + 200;

		trnData[m][0] = 1;
		trnData[m][1] = 1;
	}

	random_shuffle(trnData.begin(), trnData.end());

	// Assign Labels
	assignNandLabels(trnData, trnLabels);

	// Test Data:
	for (int i = 0; i < 1000; i++) {
		tstData[i].resize(2);
	}

	for (int j = 0; j < 200; j++) {
		tstData[j][0] = 1;
		tstData[j][1] = 1;

		int k = j + 200;

		tstData[k][0] = 0;
		tstData[k][1] = 0;

		int l = k + 200;

		tstData[l][0] = 0;
		tstData[l][1] = 1;

		int m = l + 200;

		tstData[m][0] = 1;
		tstData[m][1] = 0;
	}

	random_shuffle(tstData.begin(), tstData.end());

	// Assign Labels
	assignNandLabels(tstData, tstLabels);
}

/**
 * @brief	Create and use a neural network for the purpose of learning NAND gates.
 */

void nandGates() {
	vector<vector<double>> nandData(1000);
	vector<double> nandLabels(1000);
	vector<vector<double>> nandTestData(1000);
	vector<double> nandTestLabels(1000);

	generateNandData(nandData, nandLabels, nandTestData, nandTestLabels);

	Network net(3, 2, 1);

	net.train(nandData, nandLabels);
	net.test(nandTestData, nandTestLabels);
	net.save();
}

/**
 * @brief	Create and use a neural network for the purpose of character recognition.
 */

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
	int depth = 3;
	Network net(depth, MNistReader::TOTAL_PIXELS, 1);
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

/**
 * @brief	Main entry-point for this application.
 * 			
 * @param	argc	The number of command-line arguments provided.
 * @param	argv	An array of command-line argument strings.
 *
 * @return	Exit-code for the process - 0 for success, else an error code.
 */

int main(int argc, char *argv[])
{
	cout << DIVIDER << endl << "MNIST Character Recognition Neural Network" << endl << DIVIDER << endl;

	nandGates();
	//characterRecognition();

	string in;
	cin >> in;
}