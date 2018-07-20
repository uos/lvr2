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

LVRSpectralDialog::LVRSpectralDialog(QTreeWidget* treeWidget, QMainWindow* mainWindow, PointBufferBridgePtr points) :
   m_points(points), m_mainWindow(mainWindow)
{
    points->getSpectralChannels(m_r, m_g, m_b);

    // Setup DialogUI and events
    m_dialog = new QDialog(treeWidget);
    m_spectralDialog = new SpectralDialog;
    m_spectralDialog->setupUi(m_dialog);

    m_spectralDialog->horizontalSlider_Hyperspectral_red->setValue(m_r);
    m_spectralDialog->horizontalSlider_Hyperspectral_green->setValue(m_g);
    m_spectralDialog->horizontalSlider_Hyperspectral_blue->setValue(m_b);
    
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

    QObject::connect(m_spectralDialog->label, SIGNAL(linkActivated(const QString & link)), this, SLOT(valueChangeFinished()));
    QObject::connect(m_spectralDialog->label_2, SIGNAL(linkActivated(const QString & link)), this, SLOT(valueChangeFinished()));
    QObject::connect(m_spectralDialog->label_3, SIGNAL(linkActivated(const QString & link)), this, SLOT(valueChangeFinished()));
       
}

void LVRSpectralDialog::valueChangeFinished(){
    m_r = m_spectralDialog->horizontalSlider_Hyperspectral_red->value();
    m_g = m_spectralDialog->horizontalSlider_Hyperspectral_green->value();
    m_b = m_spectralDialog->horizontalSlider_Hyperspectral_blue->value();
    m_points->setSpectralChannels(m_r, m_g, m_b);
    m_mainWindow->resize(m_mainWindow->width(), m_mainWindow->height() + 1); //TODO: find a better way to refresh the Window pls
    m_mainWindow->resize(m_mainWindow->width(), m_mainWindow->height() - 1); // I tried update, raise, focus, ... and nothing worked
    
    m_spectralDialog->label->setText("Hyperspectral red: " + QString("%1").arg((int)m_r));
    m_spectralDialog->label_2->setText("Hyperspectral green: " + QString("%1").arg((int)m_g));
    m_spectralDialog->label_3->setText("Hyperspectral blue: " + QString("%1").arg((int)m_b));
     
}

}