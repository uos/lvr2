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
 * DataCollector.h
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef DATACOLLECTOR_H_
#define DATACOLLECTOR_H_



#include <string>
using std::string;

#include "display/Renderable.hpp"

#include "../app/Types.h"
#include "../viewers/Viewer.h"
#include "../widgets/CustomTreeWidgetItem.h"

using lssr::Renderable;
using lssr::Vertex;
using lssr::Renderable;
using lssr::BoundingBox;

class Visualizer
{
public:
	Visualizer();
	virtual ~Visualizer();
	virtual Renderable* renderable();
	virtual string	name();
	virtual BoundingBox<Vertex<float> >* boundingBox() { return m_renderable->boundingBox();}
	virtual CustomTreeWidgetItem* treeItem() { return m_treeItem;}
	virtual ViewerType supportedViewerType() {return PERSPECTIVE_VIEWER;}

protected:

	CustomTreeWidgetItem*   m_treeItem;
	Renderable*	            m_renderable;
	string		            m_name;


};

#endif /* DATACOLLECTOR_H_ */
