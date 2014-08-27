#include <QFileDialog>
#include "LVRReconstructionEstimateNormalsDialog.hpp"

namespace lvr
{

LVREstimateNormalsDialog::LVREstimateNormalsDialog(LVRPointCloudItem* pc, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* window) :
   m_pc(pc), m_parent(parent), m_treeWidget(treeWidget), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new EstimateNormalsDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVREstimateNormalsDialog::~LVREstimateNormalsDialog()
{
    // TODO Auto-generated destructor stub
}

void LVREstimateNormalsDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(estimateNormals()));
}

void LVREstimateNormalsDialog::estimateNormals()
{

}

}
