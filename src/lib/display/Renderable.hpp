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
 * Renderable.h
 *
 *  Created on: 26.08.2008
 *      Author: Thomas Wiemann
 *
 */

#ifndef RENDERABLE_H_
#define RENDERABLE_H_

#include "gl.h"
#include "glut.h"

#include <string>
using namespace std;

#include "geometry/Matrix4.hpp"
#include "geometry/Quaternion.hpp"
#include "geometry/BoundingBox.hpp"

#include "io/Model.hpp"

namespace lssr
{

class Renderable {
public:
	Renderable();
	Renderable(const Renderable &other);
	Renderable(string name);
	Renderable(Matrix4<float> m, string name);

	virtual ~Renderable();
	virtual void render() = 0;

	void setTransformationMatrix(Matrix4<float> m);

	void setName(string s){m_name = s;};
	void setVisible(bool s){m_visible = s;};
	void setRotationSpeed(float s){m_rotationSpeed = s;};
	void setTranslationSpeed(float s){m_translationSpeed = s;};
	void setActive(bool a){m_active = a;};
	void setSelected(bool s) {m_selected = s;};

	bool isActive(){return m_active;}
	bool isSelected() { return m_selected;}

	void toggle(){m_active = !m_active;}

	void moveX(bool invert = 0)
		{invert ? m_position.x -= m_translationSpeed: m_position.x += m_translationSpeed; computeMatrix();};

	void moveY(bool invert = 0)
		{invert ? m_position.y -= m_translationSpeed: m_position.y += m_translationSpeed; computeMatrix();};

	void moveZ(bool invert = 0)
		{invert ? m_position.z -= m_translationSpeed: m_position.z += m_translationSpeed; computeMatrix();};

	void rotX(bool invert = 0);
	void rotY(bool invert = 0);
	void rotZ(bool invert = 0);

	void yaw(bool invert = 0);
	void pitch(bool invert = 0);
	void roll(bool invert = 0);

	void accel(bool invert = 0);
	void lift(bool invert = 0);
	void strafe(bool invert = 0);

	void scale(float s);

	void showAxes(bool on){ m_showAxes = on;};

	void compileAxesList();

	string Name() {return m_name;};
	Matrix4<float> getTransformation(){return m_transformation;};

	BoundingBox<Vertex<float> >* boundingBox() { return m_boundingBox;};

	ModelPtr model()
    {
        return m_model;
    }

protected:

	virtual void    transform();
	void            computeMatrix();

	bool                         m_visible;
	bool                         m_showAxes;
	bool                         m_active;
	bool                         m_selected;


	int                          m_listIndex;
	int                          m_activeListIndex;
	int                          m_axesListIndex;

	float                        m_rotationSpeed;
	float                        m_translationSpeed;
	float                        m_scaleFactor;

    string                       m_name;

	Normal<float>                m_xAxis;
	Normal<float>                m_yAxis;
	Normal<float>                m_z_Axis;

	Vertex<float>                m_position;

    Matrix4<float>               m_transformation;
    BoundingBox<Vertex<float> >* m_boundingBox;

    ModelPtr                       m_model;
};

}

#endif /* RENDERABLE_H_ */
