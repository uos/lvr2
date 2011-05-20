/*
 * OrthoCam.cpp
 *
 *  Created on: 27.09.2010
 *      Author: Thomas Wiemann
 */

#include "OrthoCamera.h"
#include <iostream>

OrthoCamera::OrthoCamera() : qglviewer::Camera()
{

}

OrthoCamera::~OrthoCamera()
{
	// TODO Auto-generated destructor stub
}

float OrthoCamera::zNear() const
{
	return  0.2 * sceneRadius();
}
