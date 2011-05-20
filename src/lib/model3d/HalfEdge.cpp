/*
 * HalfEdge.cpp
 *
 *  Created on: 03.12.2008
 *      Author: twiemann
 */

#include "HalfEdge.h"


HalfEdge::HalfEdge(){
	start = end = 0;
	next = pair = 0;
	face = 0;
	used = false;
}

HalfEdge::~HalfEdge(){
	delete next;
	delete pair;
}
