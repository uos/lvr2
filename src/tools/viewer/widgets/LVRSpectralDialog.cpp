#include <QFileDialog>
#include "LVRSpectralDialog.hpp"

//#include <vtkFFMPEGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include <lvr/io/PointBuffer.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/Model.hpp>
#include <lvr/io/DataStruct.hpp>
#include <lvr/registration/ICPPointAlign.hpp>
#include "lvr/io/PointBuffer.hpp"

#include <cstring>

namespace lvr
{

LVRSpectralDialog::LVRSpectralDialog(QTreeWidget* treeWidget, PointBufferBridgePtr points) :
   m_points(points)
{
    points->getSpectralChannels(m_r, m_g, m_b);

    // Setup DialogUI and events
    QDialog* dialog = new QDialog(treeWidget);
    m_dialog = new SpectralDialog;
    m_dialog->setupUi(dialog);

    m_dialog->horizontalSlider_Hyperspectral_red->setValue(m_r);
    m_dialog->horizontalSlider_Hyperspectral_green->setValue(m_g);
    m_dialog->horizontalSlider_Hyperspectral_blue->setValue(m_b);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRSpectralDialog::~LVRSpectralDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRSpectralDialog::connectSignalsAndSlots()
{
    //QObject::connect(m_dialog->addFrame_button, SIGNAL(pressed()), this, SLOT(addFrame()));
    //QObject::connect(m_dialog->removeFrame_button, SIGNAL(pressed()), this, SLOT(removeFrame()));
    //QObject::connect(m_dialog->clearFrames_button, SIGNAL(pressed()), this, SLOT(clearFrames()));
    //QObject::connect(m_dialog->interpolation_box, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(changeInterpolation(const QString&)));
    //QObject::connect(m_dialog->savePath_button, SIGNAL(pressed()), this, SLOT(savePath()));
    //QObject::connect(m_dialog->loadPath_button, SIGNAL(pressed()), this, SLOT(loadPath()));
    //QObject::connect(m_dialog->saveVideo_button, SIGNAL(pressed()), this, SLOT(saveVideo()));
    //QObject::connect(m_dialog->play_button, SIGNAL(pressed()), this, SLOT(play()));

    QObject::connect(m_dialog->horizontalSlider_Hyperspectral_red, SIGNAL(valueChanged(int)), this, SLOT(changeSliderRed(int)));
    QObject::connect(m_dialog->horizontalSlider_Hyperspectral_blue, SIGNAL(valueChanged(int)), this, SLOT(changeSliderBlue(int)));
    QObject::connect(m_dialog->horizontalSlider_Hyperspectral_green, SIGNAL(valueChanged(int)), this, SLOT(changeSliderGreen(int)));


}

void LVRSpectralDialog::changeSliderRed(int channelRed){
    m_r = channelRed;
    m_points->setSpectralChannels(m_r, m_g, m_b);
}

void LVRSpectralDialog::changeSliderGreen(int channelGreen){
    m_g = channelGreen;
    m_points->setSpectralChannels(m_r, m_g, m_b);
}

void LVRSpectralDialog::changeSliderBlue(int channelBlue){
    m_b = channelBlue;
    m_points->setSpectralChannels(m_r, m_g, m_b);
}




}