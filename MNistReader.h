/**
 * @file	MNistReader.h.
 *
 * @brief	Declares the structure of a reader for the MNIST dataset
 */

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

/**
 * @class	MNistReader
 *
 * @brief	A reader for the MNIST dataset.
 *
 * @author	Thomas Fisher
 * @date	04/05/2017
 */

class MNistReader
{
public:
	MNistReader();
	~MNistReader();
	void readImages(int numberOfSamples, int dataOfAnImage, vector<vector<double>> &vec, string filepath);
	void readLabels(int numberOfSamples, vector<double> &vec, string filepath);

	static const int IMAGE_SIZE_PX = 28;
	static const int TOTAL_PIXELS = IMAGE_SIZE_PX * IMAGE_SIZE_PX;
	static const int TESTING_SIZE = 10000;
	static const int TRAINING_SIZE = 60000;
	const string TEST_IMAGES = "t10k-images.idx3-ubyte";
	const string TEST_LABELS = "t10k-labels.idx1-ubyte";
	const string TRAINING_IMAGES = "train-images.idx3-ubyte";
	const string TRAINING_LABELS = "train-labels.idx1-ubyte";
private:
	int reverseInt(int i);
};


