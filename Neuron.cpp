#include "Neuron.h"
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

Neuron::Neuron() :weights(0), deltaValues(0),
output(0), gain(0), weightGain(0)
{

}

Neuron::~Neuron()
{
	if (weights)
		delete[] weights;
	if (deltaValues)
		delete[] deltaValues;
}

void Neuron::create(int inputCount)
{
	assert(inputCount);
	float sign = -1;
	float random;
	weights = new float[inputCount];

	for (int i = 0; i < inputCount; i++)
	{
		random = (float(rand()) / float(RAND_MAX)) / 2.f;
		random *= sign;
		sign *= -1;
		weights[i] = random;
		deltaValues[i] = 0;
	}

	gain = 1;

	random = (float(rand()) / float(RAND_MAX)) / 2.f;
	random *= sign;
	sign *= -1;
	weightGain = random;
}