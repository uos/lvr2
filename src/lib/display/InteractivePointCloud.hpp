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
 * InteractivePointCloud.hpp
 *
 *  Created on: 02.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef INTERACTIVEPOINTCLOUD_HPP_
#define INTERACTIVEPOINTCLOUD_HPP_

#include "Renderable.hpp"

#include "io/model.hpp"

namespace lssr
{

class InteractivePointCloud: public lssr::Renderable
{
public:
	InteractivePointCloud();
	InteractivePointCloud(PointBufferPtr buffer);
	virtual ~InteractivePointCloud();

	virtual void render();

	void updateBuffer(PointBufferPtr buffer);


private:

	PointBufferPtr			m_buffer;
};

} /* namespace lssr */
#endif /* INTERACTIVEPOINTCLOUD_HPP_ */
