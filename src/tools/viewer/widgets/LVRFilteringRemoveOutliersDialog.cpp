#include <QFileDialog>
#include "LVRFilteringRemoveOutliersDialog.hpp"

namespace lvr
{

LVRRemoveOutliersDialog::LVRRemoveOutliersDialog(LVRPointCloudItem* pc, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* window) :
   m_pc(pc), m_parent(parent), m_treeWidget(treeWidget), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new RemoveOutliersDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRRemoveOutliersDialog::~LVRRemoveOutliersDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRRemoveOutliersDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(removeOutliers()));
}

void LVRRemoveOutliersDialog::removeOutliers()
{
    QDoubleSpinBox* standardDeviation_box = m_dialog->doubleSpinBox_st;
    float standardDeviation = (float)standardDeviation_box->value();
    QDoubleSpinBox* meanK_box = m_dialog->doubleSpinBox_st;
    float meanK = (float)meanK_box->value();
	/*
    PCLFiltering filter(m_pc->getPointBuffer());
    filter.applyOutlierRemoval(meanK, standardDeviation);

    PointBufferPtr pb( filter.getPointBuffer() );
    ModelPtr model( new Model( pb ) );

    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (rem. outliers)";
    m_optimizedPointCloud = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_optimizedPointCloud);
    m_optimizedPointCloud->setExpanded(true); */
}

}
