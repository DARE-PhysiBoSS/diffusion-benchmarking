#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
	if (argc != 3)
		std::cout << "a.out SIZE TILE" << std::endl;

	long size = std::stol(argv[1]);
	long tile = std::stol(argv[2]);

	std::unique_ptr<float[]> data = std::make_unique<float[]>(size * size);

	for (int i = 0; i < size * size; i++)
		data[i] = rand();

	for (int i = 1; i < size; i++)
	{
		float r = rand();
		float a_s = rand();

		for (int x = 0; x < tile; x++)
		{
			data[i * tile + x] = r * (data[i * tile + x] - a_s * data[(i - 1) * tile + x]);
		}
	}

    return (int)data[size * size - 1];
}
