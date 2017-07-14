#pragma once
class Random
{
public:
	Random();

	int nextIntExcludingTop(int bottom, int topLimit);

	int nextIntIncludingTop(int bottom, int top);

	double nextDoubleExcludingTop(double bottom, double topLimit);

	double nextDoubleIncludingTop(double bottom, double top);

};