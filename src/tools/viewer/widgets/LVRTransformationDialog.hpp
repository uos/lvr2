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
 * TransformationDialog.h
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

#ifndef TRANSFORMATIONDIALOG_H_
#define TRANSFORMATIONDIALOG_H_

#include <vtkRenderWindow.h>

#include "LVRTransformationDialogUI.h"
#include "LVRModelItem.hpp"

using Ui::TransformationDialogUI;

namespace lvr
{

class LVRTransformationDialog : public QObject
{
    Q_OBJECT

public:
    LVRTransformationDialog(LVRModelItem* parent, vtkRenderWindow* renderer);
    virtual ~LVRTransformationDialog();

public Q_SLOTS:

    void rotationXSlided(int value);
    void rotationYSlided(int value);
    void rotationZSlided(int value);

    void rotationXEntered(double value);
    void rotationYEntered(double value);
    void rotationZEntered(double value);

    void translationXEntered(double value);
    void translationYEntered(double value);
    void translationZEntered(double value);

    void stepChanged(double value);

    void reset();
    void save();
private:

    void connectSignalsAndSlots();
    void transformLocal();
    void transformGlobal();

    Pose                        m_pose;
    TransformationDialogUI*     m_dialogUI;
    LVRModelItem*               m_parent;
    vtkRenderWindow*            m_renderWindow;

};

} // namespace lvr

#endif /* TRANSFORMATIONDIALOG_H_ */
