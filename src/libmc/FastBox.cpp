/*
 * FastBox.cpp
 *
 *  Created on: 22.10.2008
 *      Author: twiemann
 */

#include "FastBox.h"

FastBox::FastBox() {
	for(int i = 0; i < 8; i++)  vertices[i]      = -1;
	for(int i = 0; i < 12; i++) intersections[i] = -1;
}

FastBox::~FastBox() {
	// TODO Auto-generated destructor stub
}
