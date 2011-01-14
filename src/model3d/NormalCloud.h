/*
 * NormalCloud.h
 *
 *  Created on: 13.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef NORMALCLOUD_H_
#define NORMALCLOUD_H_

#include <vector>
#include <string>
#include <fstream>
using namespace std;

#include "Renderable.h"

class NormalCloud : public Renderable{
public:
	NormalCloud(string Filename);
	NormalCloud(const NormalCloud &o);

	virtual inline void render();
	virtual ~NormalCloud();

private:

	void initDisplayList();

	vector<Vertex> points;
	vector<Normal> normals;


};

inline void NormalCloud::render(){
	if(visible){
		glPushMatrix();
		glMultMatrixf(transformation.getData());
		glCallList(listIndex);
		glPopMatrix();
	}
}

#endif /* NORMALCLOUD_H_ */
