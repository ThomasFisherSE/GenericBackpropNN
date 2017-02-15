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
private:
	int reverseInt(int i);
};


