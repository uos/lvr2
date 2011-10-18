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
 * CustomTreeWidgetItem.h
 *
 *  Created on: 13.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef CUSTOMTREEWIDGETITEM_H_
#define CUSTOMTREEWIDGETITEM_H_

#include <QtGui>

#include "display/Renderable.hpp"

using lssr::Renderable;

enum
{
    ServerItem = 1001,
    InterfaceItem,
    PointCloudItem,
    TriangleMeshItem,
    MultiPointCloudItem
};

enum
{
    Mesh              = 0x01,
    Points            = 0x02,
    PointNormals      = 0x04,
    VertexNormals     = 0x08,
    Vertices          = 0x10,
    Wireframe         = 0x20
};


class CustomTreeWidgetItem : public QTreeWidgetItem
{
public:
	CustomTreeWidgetItem(int type);
	CustomTreeWidgetItem(QTreeWidgetItem* parent, int type);

	virtual ~CustomTreeWidgetItem();
	virtual void setRenderable(Renderable* renderable) {m_renderable = renderable;};

	bool toggled();
	void setInitialState(Qt::CheckState state);

	Renderable* renderable() { return m_renderable;}
	void setName(string name);

    bool centerOnClick()                { return m_centerOnClick;}
    void setViewCentering(bool center)  { m_centerOnClick = center;}

    string name() { return m_name;}

    void setSupportedRenderModes(int mode) {m_renderMode = mode;}

    bool supportsMode(int mode);
protected:

	Qt::CheckState      m_oldCheckState;
	Renderable*         m_renderable;
	string              m_name;
	bool                m_centerOnClick;
	int                 m_renderMode;

};

#endif /* CUSTOMTREEWIDGETITEM_H_ */
