/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /**
 * LVRTransformationDialog.cpp
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

#include <QFileDialog>
#include "LVRTransformationDialog.hpp"

namespace lvr2
{

LVRTransformationDialog::LVRTransformationDialog(LVRModelItem* parent, vtkRenderWindow* window) :
   m_parent(parent), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(parent->treeWidget());
    m_dialogUI = new TransformationDialogUI;
    m_dialogUI->setupUi(dialog);

    m_pose = m_pose_original = parent->getPose();

    // Set slider to correct positions
    m_dialogUI->sliderXRot->setValue(m_pose.r * 100000.0);
    m_dialogUI->sliderYRot->setValue(m_pose.t * 100000.0);
    m_dialogUI->sliderZRot->setValue(m_pose.p * 100000.0);
    m_dialogUI->spinBoxXRot->setValue(m_pose.r);
    m_dialogUI->spinBoxYRot->setValue(m_pose.t);
    m_dialogUI->spinBoxZRot->setValue(m_pose.p);
    m_dialogUI->spinBoxXTrans->setValue(m_pose.x);
    m_dialogUI->spinBoxYTrans->setValue(m_pose.y);
    m_dialogUI->spinBoxZTrans->setValue(m_pose.z);
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

    QObject::connect(m_dialogUI->buttonBox, SIGNAL(rejected()), this, SLOT(restoreAndClose()));
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

void LVRTransformationDialog::restoreAndClose()
{
    m_parent->setPose(m_pose_original);
    m_renderWindow->Render();
}

void LVRTransformationDialog::reset()
{
    // Reset values
    m_pose.p = 0.0;
    m_pose.r = 0.0;
    m_pose.t = 0.0;
    m_pose.x = 0.0;
    m_pose.y = 0.0;
    m_pose.z = 0.0;

    // Reset sliders
    m_dialogUI->sliderXRot->setValue(0);
    m_dialogUI->sliderYRot->setValue(0);
    m_dialogUI->sliderZRot->setValue(0);

    // Reset spin boxes
    m_dialogUI->spinBoxXRot->setValue(0.0);
    m_dialogUI->spinBoxYRot->setValue(0.0);
    m_dialogUI->spinBoxZRot->setValue(0.0);

    m_dialogUI->spinBoxXTrans->setValue(0.0);
    m_dialogUI->spinBoxYTrans->setValue(0.0);
    m_dialogUI->spinBoxZTrans->setValue(0.0);
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


} // namespace lvr2
