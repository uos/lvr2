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
 * GroundPlane.h
 *
 *  Created on: 27.08.2008
 *      Author: Thomas Wiemann
 */

#ifndef GROUNDPLANE_H_
#define GROUNDPLANE_H_

#include "Renderable.hpp"

namespace lvr
{

class GroundPlane : public Renderable{

public:
	GroundPlane();
	GroundPlane(int increment, int count);

	virtual ~GroundPlane();
	virtual void render();
	virtual void transform(Matrix4<float> m);

private:
	void drawGrid(int increment, int count);

};

} // namespace lvr

#endif /* GROUNDPLANE_H_ */
