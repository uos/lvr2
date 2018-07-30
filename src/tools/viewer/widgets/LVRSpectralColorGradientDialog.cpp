#include <QFileDialog>
#include "LVRSpectralColorGradientDialog.hpp"

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

LVRSpectralColorGradientDialog::LVRSpectralColorGradientDialog(QTreeWidget* treeWidget, QMainWindow* mainWindow, PointBufferBridgePtr points, vtkRenderer* renderer):
   m_points(points), m_mainWindow(mainWindow), m_renderer(renderer)
{
    // Setup DialogUI and events
    m_dialog = new QDialog(treeWidget);
    m_spectralDialog = new SpectralColorGradient;
    m_spectralDialog->setupUi(m_dialog);

    // get values
    size_t n, n_channels;
    points->getPointBuffer()->getPointSpectralChannelsArray(n, n_channels);
    points->getSpectralColorGradient(m_gradient, m_gradientChannel, m_useNormalizedGradient);
    m_useNDVI = false; //TODO: implement

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

LVRSpectralColorGradientDialog::~LVRSpectralColorGradientDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRSpectralColorGradientDialog::connectSignalsAndSlots()
{
    QObject::connect(m_spectralDialog->horizontalSlider_channel, SIGNAL(valueChanged(int)), this, SLOT(updateGradientView()));
    QObject::connect(m_spectralDialog->comboBox_colorgradient, SIGNAL(currentIndexChanged(int)), this, SLOT(updateGradientView()));    
    QObject::connect(m_spectralDialog->checkBox_normcolors, SIGNAL(stateChanged(int)), this, SLOT(updateGradientView()));
    QObject::connect(m_spectralDialog->checkBox_NDVI, SIGNAL(stateChanged(int)), this, SLOT(updateGradientView()));
    QObject::connect(m_spectralDialog->pushButton_cg_apply, SIGNAL(released()), this, SLOT(setTypeGradient()));
    QObject::connect(m_spectralDialog->pushButton_cg_close, SIGNAL(released()), this, SLOT(exitDialog()));
}

void LVRSpectralColorGradientDialog::exitDialog()
{
    m_dialog->done(0);
}

void LVRSpectralColorGradientDialog::setTypeGradient()
{
    m_points->useGradient(true);
    m_renderer->GetRenderWindow()->Render();
}

void LVRSpectralColorGradientDialog::updateGradientView()
{
    m_useNormalizedGradient = m_spectralDialog->checkBox_normcolors->isChecked();
    m_useNDVI = m_spectralDialog->checkBox_NDVI->isChecked();
    m_spectralDialog->horizontalSlider_channel->setEnabled(!m_useNDVI);

    m_gradientChannel = m_spectralDialog->horizontalSlider_channel->value();
    m_gradient = (GradientType)m_spectralDialog->comboBox_colorgradient->currentIndex();
    m_points->setSpectralColorGradient(m_gradient, m_gradientChannel, m_useNormalizedGradient, m_useNDVI);

    refreshDisplays();
    m_renderer->GetRenderWindow()->Render();
}

void LVRSpectralColorGradientDialog::refreshDisplays()
{
    PointBufferPtr p = m_points->getPointBuffer();
    m_spectralDialog->label_cg_channel->setText(QString("Wavelength: %1nm").arg(p->getWavelength(m_gradientChannel)));
}

}
