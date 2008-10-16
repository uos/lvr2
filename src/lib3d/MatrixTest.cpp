/*
 * MatrixTest.cpp
 *
 *  Created on: 09.09.2008
 *      Author: Thomas Wiemann
 *
 */

#include <iostream>
using namespace std;

#include "Matrix4.h"

int main(int argc, char** argv){

	double data[] = { -0.06, 0.1469, -0.987, 0,
			          0.6584, -0.7372, -0.1497, 0,
			          -0.75, -0.6594, -0.0525, 0,
			          121.228, 1799.52, -83.0901, 1};


	Matrix4 m1(data);

	cout << "INITIAL MATRIX: " << endl;
	cout << m1 << endl;

	double pose[6];
	m1.toPostionAngle(pose);

	cout << "POSTION / ANGLE: " << endl;
	cout << pose[0] << " " << pose[1] << " " << pose[2] << endl;
	cout << pose[3] << " " << pose[4] << " " << pose[5] << endl;
	cout << endl;

	Matrix4 m2(Vertex(pose[0], pose[1], pose[2]),
			  Vertex(pose[3], pose[4], pose[5]));

	cout << "MATRIX FROM POSITION / ANGLE: " << endl;
	cout << m2 << endl;


	return 0;
}


