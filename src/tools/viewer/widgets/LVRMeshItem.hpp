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

/**
 * LVRMeshItem.h
 *
 *  @date Feb 11, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRMESHITEM_H_
#define LVRMESHITEM_H_

#include <QTreeWidgetItem>
#include <QColor>

#include "../vtkBridge/LVRMeshBufferBridge.hpp"

namespace lvr
{

class LVRMeshItem : public QTreeWidgetItem
{
public:
    LVRMeshItem(MeshBufferBridgePtr& ptr, QTreeWidgetItem* parent = 0);
    virtual ~LVRMeshItem();
    QColor	getColor();
    void    setColor(QColor &c);
    void    setSelectionColor(QColor &c);
    void    resetColor();
    float	getOpacity();
    void    setOpacity(float &opacity);
    bool	getVisibility();
    void    setVisibility(bool &visiblity);
    int     getShading();
    void    setShading(int &shader);
    MeshBufferPtr   getMeshBuffer();
    vtkSmartPointer<vtkActor> getActor();

private:
    QColor                  m_color;
    MeshBufferBridgePtr     m_meshBridge;
    float					m_opacity;
    bool					m_visible;
    int                     m_shader;

};

} /* namespace lvr */

#endif /* LVRMESHITEM_H_ */
