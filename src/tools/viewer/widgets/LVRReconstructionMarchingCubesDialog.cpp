#include <QFileDialog>
#include "LVRReconstructionMarchingCubesDialog.hpp"

namespace lvr
{

LVRReconstructViaMarchingCubesDialog::LVRReconstructViaMarchingCubesDialog(LVRModelItem* parent, vtkRenderWindow* window) :
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
    QComboBox* pcm = m_dialog->comboBox_pcm;
    QCheckBox* ransacCheckBox = m_dialog->checkBox_RANSAC;
    QDoubleSpinBox* kn = m_dialog->doubleSpinBox_kn;
    QDoubleSpinBox* kd = m_dialog->doubleSpinBox_kd;
    QDoubleSpinBox* ki = m_dialog->doubleSpinBox_ki;
    QCheckBox* reestimateNormals = m_dialog->checkBox_renormals;
    cout << "PCM: " << pcm->currentText().toStdString() << endl;
    cout << "RANSAC enabled? " << ransacCheckBox->isChecked() << endl;
    cout << "kn: " << kn->value() << endl;
    cout << "kd: " << kd->value() << endl;
    cout << "ki: " << ki->value() << endl;
    cout << "(re-)estimate normals? " << reestimateNormals->isChecked() << endl;
}

}
