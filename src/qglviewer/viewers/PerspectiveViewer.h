/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


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
