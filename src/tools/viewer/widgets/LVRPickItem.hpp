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
 * LVRPickItem.hpp
 *
 *  @date Feb 20, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRPICKITEM_HPP_
#define LVRPICKITEM_HPP_

#include <QtGui/qtreewidget.h>

#include "LVRItemTypes.hpp"
#include "../vtkBridge/LVRVtkArrow.hpp"

namespace lvr
{

class LVRPickItem: public QTreeWidgetItem
{
public:
    LVRPickItem(QTreeWidget* parent, int type = LVRPickItemType);
    virtual ~LVRPickItem();

    void setStart(double* start);
    void setEnd(double* end);
    LVRVtkArrow* getArrow();

    double* getStart();
    double* getEnd();

private:
    double*         m_start;
    double*         m_end;
    LVRVtkArrow*    m_arrow;
};

} /* namespace lvr */

#endif /* LVRPICKITEM_HPP_ */
