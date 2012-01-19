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
 * TransformationDialog.cpp
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

#include <QFileDialog>
#include "TransformationDialog.h"

TransformationDialog::TransformationDialog(QWidget* parent, Renderable* r)
    : m_renderable(r), m_parent(parent)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(parent);
    m_dialog = new TransformationDialogUI;
    m_dialog->setupUi(dialog);

    m_rotX = 0.0;
    m_rotY = 0.0;
    m_rotZ = 0.0;

    m_posX = 0.0;
    m_posY = 0.0;
    m_posY = 0.0;

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();


}

TransformationDialog::~TransformationDialog()
{
    // TODO Auto-generated destructor stub
}

void TransformationDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonReset, SIGNAL(clicked()),
             this, SLOT(reset()));


    // Slider and spin box for x rotation
    QObject::connect(m_dialog->sliderXRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationXSlided(int)));

    QObject::connect(m_dialog->spinBoxXRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationXEntered(double)));

    // Slider and spin box for y rotation
    QObject::connect(m_dialog->sliderYRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationYSlided(int)));

    QObject::connect(m_dialog->spinBoxYRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationYEntered(double)));

    // Slider and spin box for z rotation
    QObject::connect(m_dialog->sliderZRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationZSlided(int)));

    QObject::connect(m_dialog->spinBoxZRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationZEntered(double)));

    // Spin boxes for translation
    QObject::connect(m_dialog->spinBoxXTrans, SIGNAL(valueChanged(double)),
              this, SLOT(translationXEntered(double)));

    QObject::connect(m_dialog->spinBoxYTrans, SIGNAL(valueChanged(double)),
              this, SLOT(translationYEntered(double)));

    QObject::connect(m_dialog->spinBoxZTrans, SIGNAL(valueChanged(double)),
              this, SLOT(translationZEntered(double)));


    QObject::connect(m_dialog->buttonSave, SIGNAL(clicked()), this, SLOT(save()));


}

void TransformationDialog::save()
{
    QString filename = QFileDialog::getSaveFileName(m_parent, "Save tranformation to pose file", "", "*.pose");

    ofstream out(filename.toStdString().c_str());
    out << m_posX << " " << m_posY << " " << m_posZ << " " << m_rotX << " " << m_rotY << " " << m_rotZ;
    out.close();
}

void TransformationDialog::reset()
{
    // Reset values
    m_rotX = m_rotY = m_rotZ = 0.0;
    m_posX = m_posY = m_posZ = 0.0;

    // Reset sliders
    m_dialog->sliderXRot->setValue(0);
    m_dialog->sliderYRot->setValue(0);
    m_dialog->sliderZRot->setValue(0);

    // Reset spin boxes
    m_dialog->spinBoxXRot->setValue(0.0);
    m_dialog->spinBoxYRot->setValue(0.0);
    m_dialog->spinBoxZRot->setValue(0.0);

}

void TransformationDialog::transformGlobal()
{

}

void TransformationDialog::transformLocal()
{
    // Transform object
    Vertex<float> translation(m_posX, m_posY, m_posZ);
    Vertex<float> rotation(m_rotX * 0.017453293, m_rotY * 0.017453293, m_rotZ * 0.017453293);
    m_renderable->setTransformationMatrix(Matrix4<float>(translation, rotation));

    // Update rendering
    m_parent->repaint();
}

void TransformationDialog::rotationXEntered(double value)
{
    m_rotX = value;

    // Update slider
    m_dialog->sliderXRot->setValue((int)m_rotX * 100000.0);

    // Transform selected object
    transformLocal();
}

void TransformationDialog::rotationYEntered(double value)
{
    m_rotY = value;

    // Update slider
    m_dialog->sliderYRot->setValue((int)m_rotY * 100000.0);

    // Transform selected object
    transformLocal();
}

void TransformationDialog::rotationZEntered(double value)
{
    m_rotZ = value;

    // Update slider
    m_dialog->sliderZRot->setValue((int)m_rotZ * 100000.0);

    // Transform selected object
    transformLocal();
}

void TransformationDialog::translationXEntered(double value)
{
    m_posX = value;

    // Transform selected object
    transformLocal();
}

void TransformationDialog::translationYEntered(double value)
{
    m_posY = value;

    // Transform selected object
    transformLocal();
}

void TransformationDialog::translationZEntered(double value)
{
    m_posZ = value;

    // Transform selected object
    transformLocal();
}

void TransformationDialog::rotationXSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    m_rotX = new_rot;

    // Update spin box
    m_dialog->spinBoxXRot->setValue(m_rotX);

}

void TransformationDialog::rotationYSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    m_rotY = new_rot;

    // Update spin box
    m_dialog->spinBoxYRot->setValue(m_rotY);
}

void TransformationDialog::rotationZSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    m_rotZ = new_rot;

    // Update spin box
    m_dialog->spinBoxZRot->setValue(m_rotZ);
}

