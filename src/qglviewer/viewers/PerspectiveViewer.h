/*
 * PerspectiveViewer.h
 *
 *  Created on: 22.09.2010
 *      Author: Thomas Wiemann
 */

#ifndef PERSPECTIVEVIEWER_H_
#define PERSPECTIVEVIEWER_H_

#include "Viewer.h"

enum FOGTYPE { LINEAR, EXP, EXP2};

class PerspectiveViewer: public Viewer
{
public:
	PerspectiveViewer(QWidget* parent, const QGLWidget* shared = 0);
	virtual ~PerspectiveViewer();

	void setProjectionMode(ProjectionMode mode);

	virtual void draw();
	virtual void init();
	virtual ViewerType type();

	void toggleFog();
	void setFogDensity(float density);
	void setFogType(FOGTYPE f);

	void changeFogSettings();

private:
	void createBackgroundDisplayList();
	void showEntireScene();

	int 	m_backgroundDisplayList;

	qglviewer::Camera* 	m_camera[4];
	ProjectionMode		m_projectionMode;

	bool				m_showFog;
	FOGTYPE				m_fogType;
};

#endif /* PERSPECTIVEVIEWER_H_ */
