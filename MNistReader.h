#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class MNistReader
{
public:
	MNistReader();
	~MNistReader();
	void readMnist(int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr);

	static const int IMAGE_SIZE_PX = 28;
	static const int NO_OF_PX = IMAGE_SIZE_PX * IMAGE_SIZE_PX;
	static const int TEST_SAMPLES = 10000;
	static const int TRAINING_SAMPLES = 60000;
private:
	int reverseInt(int i);
};


