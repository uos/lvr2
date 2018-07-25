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

LVRSpectralDialog::LVRSpectralDialog(QTreeWidget* treeWidget, QMainWindow* mainWindow, PointBufferBridgePtr points, vtkRenderer* renderer):
   m_points(points), m_mainWindow(mainWindow), m_renderer(renderer)
{
    // Setup DialogUI and events
    m_dialog = new QDialog(treeWidget);
    m_spectralDialog = new SpectralDialog;
    m_spectralDialog->setupUi(m_dialog);

    // Channel - Tab //
    // get values
    points->getSpectralChannels(m_r, m_g, m_b);
    size_t n, n_channels;
    points->getPointBuffer()->getPointSpectralChannelsArray(n, n_channels);

    // set values
    m_spectralDialog->horizontalSlider_Hyperspectral_red->setMaximum(n_channels - 1);
    m_spectralDialog->horizontalSlider_Hyperspectral_green->setMaximum(n_channels - 1);
    m_spectralDialog->horizontalSlider_Hyperspectral_blue->setMaximum(n_channels - 1);
    m_spectralDialog->horizontalSlider_Hyperspectral_red->setValue(m_r);
    m_spectralDialog->horizontalSlider_Hyperspectral_green->setValue(m_g);
    m_spectralDialog->horizontalSlider_Hyperspectral_blue->setValue(m_b);
    m_spectralDialog->label_hred->setText("Hyperspectral red: " + QString("%1").arg(m_r * 4 + 400) + "nm");
    m_spectralDialog->label_hgreen->setText("Hyperspectral green: " + QString("%1").arg(m_g * 4 + 400) + "nm");
    m_spectralDialog->label_hblue->setText("Hyperspectral blue: " + QString("%1").arg(m_b * 4 + 400) + "nm");
    
    // Colorgradient - Tab //
    // get values
    points->getSpectralColorGradient(m_gradient, m_gradientChannel, m_useNormalizedGradient);

    // set values
    m_spectralDialog->horizontalSlider_channel->setMaximum(n_channels);
    m_spectralDialog->horizontalSlider_channel->setValue(m_gradientChannel);
    m_spectralDialog->label_cg_channel->setText("Channel: " + QString("%1").arg(m_gradientChannel));
    m_spectralDialog->comboBox_colorgradient->setCurrentIndex(m_gradient);
    m_spectralDialog->checkBox_normcolors->setChecked(m_useNormalizedGradient);
    
    connectSignalsAndSlots();

    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

LVRSpectralDialog::~LVRSpectralDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRSpectralDialog::connectSignalsAndSlots()
{
    QObject::connect(m_spectralDialog->horizontalSlider_Hyperspectral_red, SIGNAL(valueChanged(int)), this, SLOT(valueChangeFinished()));
    QObject::connect(m_spectralDialog->horizontalSlider_Hyperspectral_green, SIGNAL(valueChanged(int)), this, SLOT(valueChangeFinished()));
    QObject::connect(m_spectralDialog->horizontalSlider_Hyperspectral_blue, SIGNAL(valueChanged(int)), this, SLOT(valueChangeFinished()));

    QObject::connect(m_spectralDialog->horizontalSlider_channel, SIGNAL(valueChanged(int)), this, SLOT(updateGradientView()));
    QObject::connect(m_spectralDialog->comboBox_colorgradient, SIGNAL(currentIndexChanged(int)), this, SLOT(updateGradientView()));    
    QObject::connect(m_spectralDialog->checkBox_normcolors, SIGNAL(stateChanged(int)), this, SLOT(updateGradientView()));

    QObject::connect(m_spectralDialog->pushButton_channel_apply, SIGNAL(released()), this, SLOT(setTypeChannel()));
    QObject::connect(m_spectralDialog->pushButton_cg_apply, SIGNAL(released()), this, SLOT(setTypeGradient()));
    QObject::connect(m_spectralDialog->pushButton_channel_close, SIGNAL(released()), this, SLOT(exitDialog()));
    QObject::connect(m_spectralDialog->pushButton_cg_close, SIGNAL(released()), this, SLOT(exitDialog()));
}

void LVRSpectralDialog::exitDialog()
{
    m_dialog->done(0);
}

void LVRSpectralDialog::setTypeChannel()
{
    m_points->useGradient(false);
    m_renderer->GetRenderWindow()->Render();
}

void LVRSpectralDialog::setTypeGradient()
{
    m_points->useGradient(true);
    m_renderer->GetRenderWindow()->Render();
}

void LVRSpectralDialog::updateGradientView()
{
    m_useNormalizedGradient = m_spectralDialog->checkBox_normcolors->isChecked();
    m_gradientChannel = m_spectralDialog->horizontalSlider_channel->value();
    m_gradient = (GradientType)m_spectralDialog->comboBox_colorgradient->currentIndex();
    m_points->setSpectralColorGradient(m_gradient, m_gradientChannel, m_useNormalizedGradient);
    m_renderer->GetRenderWindow()->Render();

    m_spectralDialog->label_cg_channel->setText("Channel: " + QString("%1").arg(m_gradientChannel));
}

void LVRSpectralDialog::valueChangeFinished(){
    m_r = m_spectralDialog->horizontalSlider_Hyperspectral_red->value();
    m_g = m_spectralDialog->horizontalSlider_Hyperspectral_green->value();
    m_b = m_spectralDialog->horizontalSlider_Hyperspectral_blue->value();
    m_points->setSpectralChannels(m_r, m_g, m_b);

    m_renderer->GetRenderWindow()->Render();
    
    m_spectralDialog->label_hred->setText("Hyperspectral red: " + QString("%1").arg(m_r * 4 + 400) + "nm");
    m_spectralDialog->label_hgreen->setText("Hyperspectral green: " + QString("%1").arg(m_g * 4 + 400) + "nm");
    m_spectralDialog->label_hblue->setText("Hyperspectral blue: " + QString("%1").arg(m_b * 4 + 400) + "nm");    
}

}