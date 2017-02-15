#pragma once
class Neuron
{
public:
	Neuron();
	~Neuron();
	void create(int inputCount);

	float *weights;
	float *deltaValues;
	float output;
	float gain;
	float weightGain;

private:
	

	
};

