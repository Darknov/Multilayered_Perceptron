#include "Random.h"
#include <ctime>
#include <cstdlib>

Random::Random()
{
	srand(time(NULL));
}

int Random::nextIntExcludingTop(int bottom, int topLimit)
{
	return rand() % (topLimit);
}

int Random::nextIntIncludingTop(int bottom, int top)
{
	return rand() % (top + 1);
}

double Random::nextDoubleExcludingTop(double bottom, double top)
{
	return bottom + (((double)rand()) / ((double)(RAND_MAX)+1.0)) * (top - bottom);
}

double Random::nextDoubleIncludingTop(double bottom, double topLimit)
{
	return bottom + ((double)rand() / RAND_MAX) * (topLimit - bottom);
}
