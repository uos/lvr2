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
 * LVRModelItem.h
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRMODELITEM_H_
#define LVRMODELITEM_H_

#include "io/Model.hpp"
#include "../vtkBridge/LVRModelBridge.hpp"
#include "LVRPoseItem.hpp"

#include <QString>
#include <QColor>
#include <QTreeWidgetItem>

namespace lvr
{

class LVRModelItem : public QTreeWidgetItem
{
public:

    LVRModelItem(ModelBridgePtr bridge, QString name = "");
    LVRModelItem(const LVRModelItem& item);
    virtual ~LVRModelItem();

    Pose    getPose();
    void    setPose( Pose& pose);
    ModelBridgePtr	getModelBridge();

protected Q_SLOTS:
	void			setVisibility(bool visible);

protected:
    ModelBridgePtr  m_modelBridge;
    QString         m_name;
    LVRPoseItem*    m_poseItem;
};

} /* namespace lvr */

#endif /* LVRMODELITEM_H_ */
