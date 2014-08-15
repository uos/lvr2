#include <QFileDialog>
#include "LVRReconstructionMarchingCubesDialog.hpp"

namespace lvr
{

LVRReconstructViaMarchingCubesDialog::LVRReconstructViaMarchingCubesDialog(LVRPointCloudItem* parent, vtkRenderWindow* window) :
   m_parent(parent), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(parent->treeWidget());
    m_dialog = new ReconstructViaMarchingCubesDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRReconstructViaMarchingCubesDialog::~LVRReconstructViaMarchingCubesDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRReconstructViaMarchingCubesDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->comboBox_pcm, SIGNAL(currentIndexChanged(const QString)), this, SLOT(toggleRANSACcheckBox(const QString)));
    /*
    QObject::connect(m_dialog->doubleSpinBox_kn, SIGNAL(valueChanged(double)), this, SLOT(printAllValues()));
    QObject::connect(m_dialog->doubleSpinBox_kd, SIGNAL(valueChanged(double)), this, SLOT(printAllValues()));
    QObject::connect(m_dialog->doubleSpinBox_ki, SIGNAL(valueChanged(double)), this, SLOT(printAllValues()));
    QObject::connect(m_dialog->checkBox_renormals, SIGNAL(stateChanged(int)), this, SLOT(printAllValues()));
    */
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(printAllValues()));
}

void LVRReconstructViaMarchingCubesDialog::save()
{

}

void LVRReconstructViaMarchingCubesDialog::toggleRANSACcheckBox(const QString &text)
{
    QCheckBox* ransacCheckBox = m_dialog->checkBox_RANSAC;
    if(text == "PCL")
    {
        ransacCheckBox->setChecked(false);
        ransacCheckBox->setCheckable(false);
    }
    else
    {
        ransacCheckBox->setCheckable(true);
    }
}

void LVRReconstructViaMarchingCubesDialog::printAllValues()
{
    QComboBox* pcm_box = m_dialog->comboBox_pcm;
    string pcm = pcm_box->currentText().toStdString();
    QCheckBox* ransac_box = m_dialog->checkBox_RANSAC;
    bool ransac = ransac_box->isChecked();
    QDoubleSpinBox* kn_box = m_dialog->doubleSpinBox_kn;
    int kn = (int) kn_box->value();
    QDoubleSpinBox* kd_box = m_dialog->doubleSpinBox_kd;
    int kd = (int) kd_box->value();
    QDoubleSpinBox* ki_box = m_dialog->doubleSpinBox_ki;
    int ki = (int) ki_box->value();
    QCheckBox* reNormals_box = m_dialog->checkBox_renormals;
    bool reestimateNormals = reNormals_box->isChecked();
    cout << "PCM: " << pcm << endl;
    cout << "RANSAC enabled? " << ransac << endl;
    cout << "kn: " << kn << endl;
    cout << "kd: " << kd << endl;
    cout << "ki: " << ki << endl;
    cout << "(re-)estimate normals? " << reestimateNormals << endl;

    PointBufferPtr pc_buffer = m_parent->getPointBuffer();

    if(pcm == "STANN" || pcm == "FLANN" || pcm == "NABO")
    {
        akSurface* aks = new akSurface(pc_buffer, pcm, kn, kd, ki);

        psSurface::Ptr surface;
        surface = psSurface::Ptr(aks);

        if(ransac) aks->useRansac(true);

        if(!surface->pointBuffer()->hasPointNormals()
                        || (surface->pointBuffer()->hasPointNormals() && reestimateNormals))
            surface->calculateSurfaceNormals();
    }
}

}
