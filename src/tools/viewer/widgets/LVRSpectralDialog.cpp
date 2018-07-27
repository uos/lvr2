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
    points->getSpectralChannels(m_r, m_g, m_b, m_use_r, m_use_g, m_use_b);
    size_t n, n_channels;
    points->getPointBuffer()->getPointSpectralChannelsArray(n, n_channels);

    // set values
    m_spectralDialog->horizontalSlider_Hyperspectral_red->setMaximum(n_channels - 1);
    m_spectralDialog->horizontalSlider_Hyperspectral_green->setMaximum(n_channels - 1);
    m_spectralDialog->horizontalSlider_Hyperspectral_blue->setMaximum(n_channels - 1);
    m_spectralDialog->horizontalSlider_Hyperspectral_red->setValue(m_r);
    m_spectralDialog->horizontalSlider_Hyperspectral_green->setValue(m_g);
    m_spectralDialog->horizontalSlider_Hyperspectral_blue->setValue(m_b);
    m_spectralDialog->checkBox_hred->setChecked(m_use_r);
    m_spectralDialog->checkBox_hgreen->setChecked(m_use_g);
    m_spectralDialog->checkBox_hblue->setChecked(m_use_b);

    // Colorgradient - Tab //
    // get values
    points->getSpectralColorGradient(m_gradient, m_gradientChannel, m_useNormalizedGradient);

    // set values
    m_spectralDialog->horizontalSlider_channel->setMaximum(n_channels - 1);
    m_spectralDialog->horizontalSlider_channel->setValue(m_gradientChannel);
    m_spectralDialog->comboBox_colorgradient->setCurrentIndex(m_gradient);
    m_spectralDialog->checkBox_normcolors->setChecked(m_useNormalizedGradient);
    
    refreshDisplays();

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
    QObject::connect(m_spectralDialog->checkBox_hred, SIGNAL(stateChanged(int)), this, SLOT(valueChangeFinished()));
    QObject::connect(m_spectralDialog->checkBox_hgreen, SIGNAL(stateChanged(int)), this, SLOT(valueChangeFinished()));  
    QObject::connect(m_spectralDialog->checkBox_hblue, SIGNAL(stateChanged(int)), this, SLOT(valueChangeFinished()));  

    QObject::connect(m_spectralDialog->horizontalSlider_channel, SIGNAL(valueChanged(int)), this, SLOT(updateGradientView()));
    QObject::connect(m_spectralDialog->comboBox_colorgradient, SIGNAL(currentIndexChanged(int)), this, SLOT(updateGradientView()));    
    QObject::connect(m_spectralDialog->checkBox_normcolors, SIGNAL(stateChanged(int)), this, SLOT(updateGradientView()));

    QObject::connect(m_spectralDialog->pushButton_channel_apply, SIGNAL(released()), this, SLOT(setTypeChannel()));
    QObject::connect(m_spectralDialog->pushButton_cg_apply, SIGNAL(released()), this, SLOT(setTypeGradient()));
    QObject::connect(m_spectralDialog->pushButton_channel_close, SIGNAL(released()), this, SLOT(exitDialog()));
    QObject::connect(m_spectralDialog->pushButton_cg_close, SIGNAL(released()), this, SLOT(exitDialog()));
    QObject::connect(m_spectralDialog->pushButton_h_close, SIGNAL(released()), this, SLOT(exitDialog()));
    QObject::connect(m_spectralDialog->pushButton_h_show, SIGNAL(released()), this, SLOT(showhistogram()));
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

    refreshDisplays();
    m_renderer->GetRenderWindow()->Render();
}

void LVRSpectralDialog::refreshDisplays()
{
    m_spectralDialog->horizontalSlider_Hyperspectral_red->setEnabled(m_use_r);
    m_spectralDialog->horizontalSlider_Hyperspectral_green->setEnabled(m_use_g);
    m_spectralDialog->horizontalSlider_Hyperspectral_blue->setEnabled(m_use_b);

    PointBufferPtr p = m_points->getPointBuffer();

    m_spectralDialog->label_hred->setText(QString("Hyperspectral red: %1nm").arg(p->getWavelength(m_r)));
    m_spectralDialog->label_hgreen->setText(QString("Hyperspectral green: %1nm").arg(p->getWavelength(m_g)));
    m_spectralDialog->label_hblue->setText(QString("Hyperspectral blue: %1nm").arg(p->getWavelength(m_b)));

    m_spectralDialog->label_cg_channel->setText(QString("Wavelength: %1nm").arg(p->getWavelength(m_gradientChannel)));
}

void LVRSpectralDialog::valueChangeFinished(){
    m_r = m_spectralDialog->horizontalSlider_Hyperspectral_red->value();
    m_g = m_spectralDialog->horizontalSlider_Hyperspectral_green->value();
    m_b = m_spectralDialog->horizontalSlider_Hyperspectral_blue->value();
    
    m_use_r = m_spectralDialog->checkBox_hred->isChecked();
    m_use_g = m_spectralDialog->checkBox_hgreen->isChecked();
    m_use_b = m_spectralDialog->checkBox_hblue->isChecked();
    
    m_points->setSpectralChannels(m_r, m_g, m_b, m_use_r, m_use_g, m_use_b);

    refreshDisplays();
    m_renderer->GetRenderWindow()->Render();
}

void LVRSpectralDialog::showhistogram()
{
    m_histogram=new LVRHistogram();
    
    m_histogram->setPointBuffer(m_points->getPointBuffer());
    m_histogram->sethistogram();
}

}
