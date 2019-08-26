/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#if _MSC_VER
#include <Windows.h>
#endif


#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glut.h>
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif

#include <string>
using namespace std;

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/geometry/Quaternion.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

#include "lvr2/io/Model.hpp"

namespace lvr2
{

class Renderable {

public:

    using Vec = BaseVector<float>;

	Renderable();
	Renderable(const Renderable &other);
	Renderable(string name);
	Renderable(Matrix4<Vec> m, string name);


	virtual ~Renderable();
	virtual void render() = 0;

	void setTransformationMatrix(Matrix4<Vec> m);

	virtual void setName(string s){m_name = s;};
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
	Matrix4<Vec> getTransformation(){return m_transformation;};

	BoundingBox<Vec>* boundingBox() { return m_boundingBox;};

	virtual ModelPtr model()
    {
        return m_model;
    }

	void setPointSize(float size)   { m_pointSize = size;}
	void setLineWidth(float width)  { m_lineWidth = width;}

	float lineWidth() { return m_lineWidth;}
	float pointSize() { return m_pointSize;}

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

	Normal<typename Vec::CoordType>       			 m_xAxis;
	Normal<typename Vec::CoordType>                  m_yAxis;
	Normal<typename Vec::CoordType>                  m_z_Axis;

	Vec                 		 m_position;

    Matrix4<Vec>                 m_transformation;
    BoundingBox<Vec>*            m_boundingBox;

    ModelPtr                     m_model;

    float                        m_lineWidth;
    float                        m_pointSize;
};

} // namespace lvr2

#endif /* RENDERABLE_H_ */
