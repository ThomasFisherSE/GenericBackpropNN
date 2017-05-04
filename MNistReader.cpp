/**
 * @file	MNistReader.cpp.
 * @author	Thomas Fisher
 * @date	04/05/2017
 * @brief	Implements a reader for the MNIST dataset
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "MNistReader.h"

using namespace std;

/**
 * @brief	Default constructor.
 */

MNistReader::MNistReader()
{

}

/**
 * @brief	Destructor.
 */

MNistReader::~MNistReader()
{

}

/**
 * @brief	Reverse int.
 * 			
 * @param	i	int value to be reversed.
 *
 * @return	Reversed int
 */

int MNistReader::reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

/**
 * @brief	Reads labels from the MNIST dataset.
 *
 * @param 		  	numberOfSamples	Number of samples.
 * @param [in,out]	vec			   	The vector to put labels into.
 * @param 		  	filename	   	Filename of the labels file.
 */

void MNistReader::readLabels(int numberOfSamples, vector<double> &vec, string filename)
{
	// Resize the vector that the data will be loaded into
	vec.resize(numberOfSamples);

	//Choose and open file to read from
	ifstream file(filename, ios::binary);

	if (file.is_open())
	{
		int magicNumber = 0;
		int numberOfImages = 0;
		file.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = reverseInt(magicNumber);
		file.read((char*)&numberOfImages, sizeof(numberOfImages));
		numberOfImages = reverseInt(numberOfImages);

		for (int i = 0; i < numberOfImages; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec[i] = (double)temp;
		}
	}
}

/**
 * @brief	Reads images from the MNIST dataset.
 *
 * @param 		  	numberOfSamples	Number of samples.
 * @param 		  	dataOfAnImage  	The amount of data in an image (i.e. 784 for MNIST images)
 * @param [in,out]	vec			   	The vector to put images into.
 * @param 		  	filepath	   	The filepath of the images file.
 */

void MNistReader::readImages(int numberOfSamples, int dataOfAnImage, vector<vector<double>> &vec, string filepath)
{
	//Counter to keep track of successful writes to array
	int successCount = 0;

	//Resize the vector that the data will be loaded into 
	vec.resize(numberOfSamples, vector<double>(dataOfAnImage));

	//Choose and open file to read from
	ifstream file(filepath, ios::binary);
	if (file.is_open())
	{
		int magicNumber = 0;
		int numberOfImages = 0;
		int rows = 0;
		int cols = 0;
		file.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = reverseInt(magicNumber);
		file.read((char*)&numberOfImages, sizeof(numberOfImages));
		numberOfImages = reverseInt(numberOfImages);
		file.read((char*)&rows, sizeof(rows));
		rows = reverseInt(rows);
		file.read((char*)&cols, sizeof(cols));
		cols = reverseInt(cols);
		for (int i = 0; i<numberOfImages; ++i)
		{
			for (int r = 0; r<rows; ++r)
			{
				for (int c = 0; c<cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					vec[i][(rows*r) + c] = (double)temp;
				}
			}

			successCount += dataOfAnImage;
			cout << (int)(successCount / TOTAL_PIXELS / (double)numberOfSamples*100.0) << "%\r";
		}
		cout << "Successfully stored " << successCount << " values into array." << endl;
		cout << "Loaded " << successCount / dataOfAnImage << " samples." << endl;
	}
}