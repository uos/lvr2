/*
 * PointCloud.h
 *
 *  Created on: 02.09.2008
 *      Author: twiemann
 */

#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "Renderable.h"
#include "BaseVertex.h"

#include <vector>
#include <string>
#include <fstream>

using namespace std;

class PointCloud : public Renderable{
public:
	PointCloud(string filename);
	virtual ~PointCloud();
	virtual inline void render();

	vector<Vertex> getPoints(){return points;};

private:
	void initDisplayList();
	vector<Vertex> points;
};

inline void PointCloud::render(){
	if(visible){
		glPushMatrix();
		glMultMatrixd(transformation.getData());
		if(show_axes) glCallList(axesListIndex);
		glDisable(GL_LIGHTING);
		if(active){
			glColor3f(1.0, 0.0, 0.0);
		} else {
			glColor3f(0.0, 0.9, 0.0);
		}
		glCallList(listIndex);
		glPopMatrix();
	}

}

#endif /* POINTCLOUD_H_ */
