/*
 * OrthoCam.h
 *
 *  Created on: 27.09.2010
 *      Author: Thomas Wiemann
 */

#ifndef ORTHOCAM_H_
#define ORTHOCAM_H_

#include <QGLViewer/qglviewer.h>

class OrthoCamera : public qglviewer::Camera
{
public:
	OrthoCamera();
	virtual ~OrthoCamera();

	virtual float zNear() const;
};

#endif /* ORTHOCAM_H_ */
