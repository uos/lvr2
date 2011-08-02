/**
 * TransformationDialog.cpp
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

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


    QObject::connect(m_dialog->sliderXRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationXSlided(int)));

    QObject::connect(m_dialog->spinBoxXRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationXEntered(double)));

    QObject::connect(m_dialog->sliderYRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationYSlided(int)));

    QObject::connect(m_dialog->spinBoxYRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationYEntered(double)));

    QObject::connect(m_dialog->sliderZRot, SIGNAL(sliderMoved(int)),
            this, SLOT(rotationZSlided(int)));

    QObject::connect(m_dialog->spinBoxZRot, SIGNAL(valueChanged(double)),
            this, SLOT(rotationZEntered(double)));

}

void TransformationDialog::reset()
{
    // Reset values
    m_rotX = m_rotY = m_rotZ = 0.0;
    m_posX = m_posX = m_posZ = 0.0;

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
    Vertex translation(0.0, 0.0, 0.0);
    Vertex rotation(m_rotX * 0.017453293, m_rotY * 0.017453293, m_rotZ * 0.017453293);
    m_renderable->setTransformationMatrix(Matrix4(translation, rotation));

    // Update rendering
    m_parent->repaint();
}

void TransformationDialog::rotationXEntered(double value)
{
    double step = m_rotX - value;
    m_rotX = value;

    // Update slider
    m_dialog->sliderXRot->setValue((int)m_rotX * 100000.0);

    // Transform selected object
    transformLocal();
}

void TransformationDialog::rotationYEntered(double value)
{
    double step = m_rotY - value;
    m_rotY = value;

    // Update slider
    m_dialog->sliderYRot->setValue((int)m_rotY * 100000.0);

    // Transform selected object
    transformLocal();
}

void TransformationDialog::rotationZEntered(double value)
{
    double step = m_rotZ - value;
    m_rotZ = value;

    // Update slider
    m_dialog->sliderZRot->setValue((int)m_rotZ * 100000.0);

    // Transform selected object
    transformLocal();
}

void TransformationDialog::rotationXSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    double step = m_rotX - new_rot;
    m_rotX = new_rot;

    // Update spin box
    m_dialog->spinBoxXRot->setValue(m_rotX);

}

void TransformationDialog::rotationYSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    double step = m_rotY - new_rot;
    m_rotY = new_rot;

    // Update spin box
    m_dialog->spinBoxYRot->setValue(m_rotY);
}

void TransformationDialog::rotationZSlided(int value)
{
    // Calc new value
    double new_rot = (double)value / 100000.0;
    double step = m_rotZ - new_rot;
    m_rotZ = new_rot;

    // Update spin box
    m_dialog->spinBoxZRot->setValue(m_rotZ);
}

