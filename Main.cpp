/**
 * @file	Main.cpp.
 * @author	Thomas Fisher
 * @date	04/05/2017
 * @brief	Implements the main class. The entry-point of the application.
 */

#include "MNistReader.h"
#include "Network.h"
#include <iostream>

using namespace std;

const string DIVIDER = "**********************************************************************************";

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
 * @brief	Normalize pixel data to be between 0 and 1.
 *
 * @param [in,out]	data	The data to be normalized.
 */

void normalizePixelData(vector<vector<double>> &data) {
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[0].size(); j++) {
			data[i][j] /= 255;
		}
	}
}

/**
 * @brief	Convert a string to an integer.
 *
 * @param	s	The string to be converted.
 *
 * @return The converted integer value.
 */

int stringToInt(string s) {
	int intVal;
	bool valid = false;

	try {
		intVal = stoi(s);
	}
	catch (...) {
		cout << "Please enter a valid integer." << endl;
		return -1;
	}

	return intVal;
}

/**
 * @brief	Convert a string to double.
 *
 * @param	s	The string.
 *
 * @return	The converted double value.
 */

double stringToDouble(string s) {
	double doubleVal;
	bool valid = false;

	try {
		doubleVal = stod(s);
	}
	catch (...) {
		cout << "Please enter a valid double." << endl;
		return -1.0;
	}

	return doubleVal;
}

/**
 * @brief	Choose options for the network.
 * 			
 * @param [in,out]	net	The net.
 */

void chooseOptions(Network &net) {
	string input;
	double doubleInput;
	int intInput;
	bool valid = false;

	cout << DIVIDER << endl;
	cout << "Choose the options for the network." << endl;

	do {
		cout << DIVIDER << endl;
		cout << "Learning Rate (Between 0 and 1. Recommended = 0.1): ";
		cin >> input;
		doubleInput = stringToDouble(input);
	
		if (doubleInput > 0 && doubleInput < 1) {
			valid = true;
			net.setEta(doubleInput);
		}
		else {
			cout << "Learning rate must be a double between 0 and 1." << endl;
			valid = false;
		}
	} while (!valid);

	do {
		cout << "Change in error required for validation check (Recommended = 0.0001): ";
		cin >> input;
		doubleInput = stringToDouble(input);

		if (doubleInput > 0) {
			valid = true;
			net.setMaxErrorChange(doubleInput);
		}
		else {
			cout << "Max error change must be a double greater than 0." << endl;
			cout << DIVIDER << endl;
			valid = false;
		}
	} while (!valid);

	do {
		cout << "Max number of epochs: (Recommended = 100): ";
		cin >> input;
		intInput = stringToInt(input);

		if (intInput > 0) {
			valid = true;
			net.setMaxEpochs(intInput);
		}
		else {
			cout << "Max number of epochs must be an integer greater than 0." << endl;
			cout << DIVIDER << endl;
			valid = false;
		}
	} while (!valid);
}

/**
 * @brief	Choose number of layers for a network.
 *
 * @return	An int, the number of layers.
 */

int chooseDepth() {
	string input;
	int intInput = 3;
	bool valid = false;

	do {
		cout << DIVIDER << endl;
		cout << "Depth of Network: ";
		cin >> input;
		intInput = stringToInt(input);

		if (intInput > 1) {
			valid = true;
			return intInput;
		}
		else {
			cout << "Depth must be an integer larger than 1." << endl;
			valid = false;
		}
	} while (!valid);

	return intInput;
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

	int depth = chooseDepth();
	Network net(depth, 2, 1);
	chooseOptions(net);

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

	normalizePixelData(testingImages);

	cout << DIVIDER << endl;

	cout << "Reading test labels..." << endl;
	dataReader.readLabels(MNistReader::TESTING_SIZE, testLabels, dataReader.TEST_LABELS);

	// Change labels to only show whether a digit is a zero or not
	for (int i = 0; i < testLabels.size(); i++) {
		if (testLabels[i] != 0) {
			testLabels[i] = 1;
		}
	}

	cout << DIVIDER << endl;

	cout << "Creating network..." << endl;
	int depth = chooseDepth();
	Network net(depth, MNistReader::TOTAL_PIXELS, 1);
	cout << "Created successfully." << endl;
	chooseOptions(net);

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
 * @brief	Program menu to determine which program to run.
 */

void programMenu() {
	string input;
	int intInput;
	bool valid = false;

	do {
		cout << "Which function would you like to train the network to perform?" << endl;
		cout << "1: NAND Gate" << endl;
		cout << "2: Handwritten Digit Recognition" << endl;
		cout << DIVIDER << endl;
		cout << "Enter '1' or '2': ";

		cin >> input;

		intInput = stringToInt(input);

		switch (intInput) {
			case 1:
				valid = true;
				nandGates();
				break;
			case 2:
				valid = true;
				characterRecognition();
				break;
			default:
				valid = false;
		}

		if (!valid) {
			cout << "Please enter either '1' or '2'." << endl;
		}

		cout << DIVIDER << endl;
	} while (!valid);
}

/**
 * @brief	Determines if program can terminate.
 *
 * @return	True if user is finished. False if otherwise.
 */

bool checkDone() {
	bool valid = false;
	string input;
	int intInput;

	do {
		cout << "1: Train and test a new network" << endl;
		cout << "2: Exit Program" << endl;
		cout << DIVIDER << endl;
		cout << "Enter '1' or '2': ";
		cin >> input;

		intInput = stringToInt(input);

		switch (intInput) {
			case 1:
				return false;
			case 2:
				return true;
			default:
				valid = false;
		}

		if (!valid) {
			cout << "Please enter either '1' or '2'." << endl;
		}
	} while (!valid);

	return true;
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
	bool done = false;

	do {
		programMenu();
		cout << "Network training and testing was successful." << endl;
		cout << DIVIDER << endl;
		done = checkDone();
	} while (!done);
}