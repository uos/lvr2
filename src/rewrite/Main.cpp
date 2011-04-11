#include "PLYIO.hpp"
#include "PLYElement.hpp"

#include <iostream>

int main(int argc, char** argv)
{
	lssr::PLYElement e("vertex", 10);
	e.addProperty("x", "float");
	e.addProperty("y", "double");
	e.addProperty("indices", "uint", "int", 10);
	e.printProperties(std::cout);

	return 0;
}
