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
 * LVRTransformationDialog.cpp
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

#include <QFileDialog>
#include "LVRTransformationDialog.hpp"

namespace lvr
{

LVRTransformationDialog::LVRTransformationDialog(LVRModelItem* parent, vtkRenderWindow* window) :
   m_parent(parent), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(parent->treeWidget());
    m_dialogUI = new TransformationDialogUI;
    m_dialogUI->setupUi(dialog);

    m_pose = parent->getPose();

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();


}

LVRTransformationDialog::~LVRTransformationDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRTransformationDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialogUI->buttonReset, SIGNAL(clicked()),
             this, SLOT(reset()));


    // Slider and spin box for x rotation
    QObject::connect(m_dialogUI->sliderXRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationXSlided(int)));

    QObject::connect(m_dialogUI->spinBoxXRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationXEntered(double)));

    // Slider and spin box for y rotation
    QObject::connect(m_dialogUI->sliderYRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationYSlided(int)));

    QObject::connect(m_dialogUI->spinBoxYRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationYEntered(double)));

    // Slider and spin box for z rotation
    QObject::connect(m_dialogUI->sliderZRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationZSlided(int)));

    QObject::connect(m_dialogUI->spinBoxZRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationZEntered(double)));

    // Spin boxes for translation
    QObject::connect(m_dialogUI->spinBoxXTrans, SIGNAL(valueChanged(double)),
              this, SLOT(translationXEntered(double)));

    QObject::connect(m_dialogUI->spinBoxYTrans, SIGNAL(valueChanged(double)),
              this, SLOT(translationYEntered(double)));

    QObject::connect(m_dialogUI->spinBoxZTrans, SIGNAL(valueChanged(double)),
              this, SLOT(translationZEntered(double)));

    QObject::connect(m_dialogUI->spinBoxStep, SIGNAL(valueChanged(double)),
              this, SLOT(stepChanged(double)));


    QObject::connect(m_dialogUI->buttonSave, SIGNAL(clicked()), this, SLOT(save()));


}

void LVRTransformationDialog::stepChanged(double value)
{
    m_dialogUI->spinBoxXTrans->setSingleStep(value);
    m_dialogUI->spinBoxYTrans->setSingleStep(value);
    m_dialogUI->spinBoxZTrans->setSingleStep(value);
}

void LVRTransformationDialog::save()
{
    QString filename = QFileDialog::getSaveFileName(m_parent->treeWidget(), "Save transformation to pose file", "", "*.pose");

    ofstream out(filename.toStdString().c_str());
    out << m_pose.x << " " << m_pose.y << " " << m_pose.z << " " << m_pose.r << " " << m_pose.t << " " << m_pose.p;
    out.close();
}

void LVRTransformationDialog::reset()
{
    // Reset values
    m_pose = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Reset sliders
    m_dialogUI->sliderXRot->setValue(0);
    m_dialogUI->sliderYRot->setValue(0);
    m_dialogUI->sliderZRot->setValue(0);

    // Reset spin boxes
    m_dialogUI->spinBoxXRot->setValue(0.0);
    m_dialogUI->spinBoxYRot->setValue(0.0);
    m_dialogUI->spinBoxZRot->setValue(0.0);

}

void LVRTransformationDialog::transformGlobal()
{

}

void LVRTransformationDialog::transformLocal()
{
    m_parent->setPose(m_pose);
    m_renderWindow->Render();
}

void LVRTransformationDialog::rotationXEntered(double value)
{
    m_pose.r = value;

    // Update slider
    m_dialogUI->sliderXRot->setValue((int)m_pose.r * 100000.0);

    // Transform selected object
    transformLocal();
}

void LVRTransformationDialog::rotationYEntered(double value)
{
    m_pose.t = value;

    // Update slider
    m_dialogUI->sliderYRot->setValue((int)m_pose.t * 100000.0);

    // Transform selected object
    transformLocal();
}

void LVRTransformationDialog::rotationZEntered(double value)
{
    m_pose.p = value;

    // Update slider
    m_dialogUI->sliderZRot->setValue((int)m_pose.p * 100000.0);

    // Transform selected object
    transformLocal();
}

void LVRTransformationDialog::translationXEntered(double value)
{
    m_pose.x = value;

    // Transform selected object
    transformLocal();
}

void LVRTransformationDialog::translationYEntered(double value)
{
    m_pose.y = value;

    // Transform selected object
    transformLocal();
}

void LVRTransformationDialog::translationZEntered(double value)
{
    m_pose.z = value;

    // Transform selected object
    transformLocal();
}

void LVRTransformationDialog::rotationXSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    m_pose.r = new_rot;

    // Update spin box
    m_dialogUI->spinBoxXRot->setValue(m_pose.r);

}

void LVRTransformationDialog::rotationYSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    m_pose.t = new_rot;

    // Update spin box
    m_dialogUI->spinBoxYRot->setValue(m_pose.t);
}

void LVRTransformationDialog::rotationZSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    m_pose.p = new_rot;

    // Update spin box
    m_dialogUI->spinBoxZRot->setValue(m_pose.p);
}

} // namespace lvr
