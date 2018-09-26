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
 * LVRTreeWidgetHelper.hpp
 *
 *  @date Apr 10, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRTREEWIDGETHELPER_HPP_
#define LVRTREEWIDGETHELPER_HPP_

#include <QTreeWidget>
#include "../widgets/LVRItemTypes.hpp"
#include "../widgets/LVRModelItem.hpp"

#include <lvr2/io/MeshBuffer2.hpp>
#include <lvr2/io/PointBuffer2.hpp>

namespace lvr2
{

class LVRTreeWidgetHelper
{
public:
    LVRTreeWidgetHelper(QTreeWidget* widget);
    virtual ~LVRTreeWidgetHelper() {};

    PointBuffer2Ptr     getPointBuffer(QString name);
    MeshBuffer2Ptr      getMeshBuffer(QString name);
    LVRModelItem*       getModelItem(QString name);


private:
    QTreeWidget*        m_treeWidget;

};

} /* namespace lvr2 */

#endif /* LVRTREEWIDGETHELPER_HPP_ */
