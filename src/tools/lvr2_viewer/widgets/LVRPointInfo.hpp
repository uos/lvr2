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

#ifndef POINTINFO_H_
#define POINTINFO_H_

#include <lvr2/io/PointBuffer2.hpp>
#include <QtWidgets>
#include <QDialog>
#include "ui_LVRPointInfo.h"
#include "LVRPlotter.hpp"

using Ui::PointInfo;

namespace lvr2
{

class LVRPointInfo : public QDialog
{
    Q_OBJECT

public:
    LVRPointInfo(QWidget* parent, PointBuffer2Ptr points, int pointId);
    virtual ~LVRPointInfo();

public Q_SLOTS:
    void refresh();
    
private:
    PointInfo      m_pointInfo;
    floatArr       m_data;
    size_t         m_numChannels;
};

} // namespace lvr2

#endif /* POINTINFO_H_ */
