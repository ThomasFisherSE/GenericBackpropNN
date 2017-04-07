#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "MNistReader.h"

using namespace std;

MNistReader::MNistReader()
{

}

MNistReader::~MNistReader()
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

// Copied
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
		int n_rows = 0;
		int n_cols = 0;
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

void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr)
{

}

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
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);
		for (int i = 0; i<number_of_images; ++i)
		{
			for (int r = 0; r<n_rows; ++r)
			{
				for (int c = 0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					vec[i][(n_rows*r) + c] = (double)temp;
				}
			}

			successCount += dataOfAnImage;
			cout << (int)(successCount / TOTAL_PIXELS / (double)numberOfSamples*100.0) << "%\r";
		}
		cout << "Successfully stored " << successCount << " values into array." << endl;
		cout << "Loaded " << successCount / dataOfAnImage << " samples." << endl;
	}
}