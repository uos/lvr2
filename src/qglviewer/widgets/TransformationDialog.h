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

#include "TransformationDialogUI.h"

#include "display/Renderable.hpp"

using lvr::Renderable;
using lvr::Vertex;
using lvr::Matrix4;

using Ui::TransformationDialogUI;

class TransformationDialog : public QObject
{
    Q_OBJECT

public:
    TransformationDialog(QWidget* parent, Renderable* r);
    virtual ~TransformationDialog();

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

    double                      m_rotX;
    double                      m_rotY;
    double                      m_rotZ;

    double                      m_posX;
    double                      m_posY;
    double                      m_posZ;

    Renderable*                 m_renderable;
    TransformationDialogUI*     m_dialog;
    QWidget*                    m_parent;

};

#endif /* TRANSFORMATIONDIALOG_H_ */
