#include <QFileDialog>
#include "LVRFilteringMLSProjectionDialog.hpp"

namespace lvr
{

LVRMLSProjectionDialog::LVRMLSProjectionDialog(LVRPointCloudItem* pc, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* window) :
   m_pc(pc), m_parent(parent), m_treeWidget(treeWidget), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new MLSProjectionDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRMLSProjectionDialog::~LVRMLSProjectionDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRMLSProjectionDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(applyMLSProjection()));
}

void LVRMLSProjectionDialog::applyMLSProjection()
{
    QDoubleSpinBox* maximumDistance_box = m_dialog->doubleSpinBox_md;
    float maximumDistance = (float)maximumDistance_box->value();

   /* PCLFiltering filter(m_pc->getPointBuffer());
    filter.applyMLSProjection(maximumDistance);

    PointBufferPtr pb( filter.getPointBuffer() );
    ModelPtr model( new Model( pb ) );

    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (post-MLS)";
    m_optimizedPointCloud = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_optimizedPointCloud);
    m_optimizedPointCloud->setExpanded(true); */
}

}
