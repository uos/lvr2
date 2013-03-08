#include <iostream>
#include <vector>
#include "KdTree.hpp"
/**
 * @brief   Main entry point for the LSSR surface executable
 */
using namespace lssr;

#include "geometry/ColorVertex.hpp"

typedef ColorVertex<float, unsigned char>      cVertex;
typedef KdTree<cVertex>                        kd;


int main(int argc, char** argv)
{
	std::cout << "!!!!!!!!!!!!!!Eigene Main gestartet!!!!!!!!!!!!!!!!!!!!" << endl;

	kd tmp;

	std::cout << "!!!!!!!!!!!!!!Programm ist durchgelaufen, scan1 bis n müssten zur Verfuegung stehen!!!!!!!!!!!!!!!!!!!!" << endl;
}
