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
 * LVRPointCloudItem.hpp
 *
 *  @date Feb 7, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRPOINTCLOUDITEM_HPP_
#define LVRPOINTCLOUDITEM_HPP_

#include "../vtkBridge/LVRPointBufferBridge.hpp"

#include <QTreeWidgetItem>
#include <QColor>

namespace lvr
{

class LVRPointCloudItem : public QTreeWidgetItem
{
public:

    LVRPointCloudItem(PointBufferBridgePtr& ptr, QTreeWidgetItem* parent = 0);
    virtual ~LVRPointCloudItem();
    void    setColor(QColor &c);
    void    setSelectionColor(QColor &c);
    void    resetColor();
    void    setPointSize(int &pointSize);
    void    setOpacity(float &opacity);
    PointBufferPtr getPointBuffer();
    vtkSmartPointer<vtkActor> getActor();

protected:
    QTreeWidgetItem*        m_parent;
    PointBufferBridgePtr    m_pointBridge;
    QColor                  m_color;
    int						m_pointSize;
    float					m_opacity;
};

} /* namespace lvr */

#endif /* LVRPOINTCLOUDITEM_HPP_ */
