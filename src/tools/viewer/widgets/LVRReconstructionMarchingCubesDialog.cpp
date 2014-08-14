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

}

void LVRReconstructViaMarchingCubesDialog::save()
{

}

}
