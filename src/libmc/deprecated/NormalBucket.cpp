/*
 * NormalBucket.cpp
 *
 *  Created on: 10.03.2009
 *      Author: twiemann
 */

#include "NormalBucket.h"

NormalBucket::NormalBucket(const NormalBucket &o){
	d = o.d;
	representative = o.representative;
	normals = o.normals;
	vertices = o.vertices;
}

NormalBucket::NormalBucket(Normal n, Vertex v){

	representative = n;
	normals.push_back(n);
	d = n * v;
	vertices.push_back(v);

}

NormalBucket::~NormalBucket(){

}

bool NormalBucket::insert(Normal n, Vertex v){

	//cout << n * representative << " / " <<  NormalBucket::epsilon << endl;

	if(n * representative > NormalBucket::epsilon){
		normals.push_back(n);
		vertices.push_back(v);
		representative += n;
		//representative.normalize();
		return true;
	}
	representative.normalize();

	return false;
}
