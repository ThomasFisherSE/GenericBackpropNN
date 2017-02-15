#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "MNistReader.h"

using namespace std;

MNistReader::DataReader()
{

}

MNistReader::~DataReader()
{

}

int MNistReader::reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNistReader::readMnist(int numberOfSamples, int dataOfAnImage, vector<vector<double>> &array)
{
	//Counter to keep track of successful writes to array
	int successCount = 0;

	//Resize array so that it contains numberOfSamples elements
	array.resize(numberOfSamples, vector<double>(dataOfAnImage));

	//Choose and open file to read from
	ifstream file("t10k-images.idx3-ubyte", ios::binary);

	if (file.is_open())
	{
		int magicNumber = 0;
		int numberOfImages = 0;
		int nRows = 0;
		int nColumns = 0;

		file.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = reverseInt(magicNumber);

		file.read((char*)&numberOfImages, sizeof(numberOfImages));
		numberOfImages = reverseInt(numberOfImages);

		file.read((char*)&nRows, sizeof(nRows));
		nRows = reverseInt(nRows);

		file.read((char*)&nColumns, sizeof(nColumns));
		nColumns = reverseInt(nColumns);
		unsigned char *temp = new unsigned char[dataOfAnImage];
		for (int i = 0; i<numberOfImages; ++i)
		{
			file.read((char*)temp, dataOfAnImage);
			for (int j = 0; j<dataOfAnImage; j++)
				array[i][j] = (double)temp[j];
			successCount += dataOfAnImage;
		}
		delete[] temp;
		cout << "Successfully stored " << successCount << " values into array." << endl;
		cout << "Loaded " << successCount / dataOfAnImage << " samples." << endl;
	}
}