#include <QFileDialog>
#include "LVROptimizationRemoveArtifactsDialog.hpp"

namespace lvr
{

LVRRemoveArtifactsDialog::LVRRemoveArtifactsDialog(LVRMeshItem* mesh, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* window) :
   m_mesh(mesh), m_parent(parent), m_treeWidget(treeWidget), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new RemoveArtifactsDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRRemoveArtifactsDialog::~LVRRemoveArtifactsDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRRemoveArtifactsDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(removeArtifacts()));
}

void LVRRemoveArtifactsDialog::removeArtifacts()
{
    QSpinBox* removeDanglingArtifacts_box = m_dialog->spinBox_rda;
    int removeDanglingArtifacts = removeDanglingArtifacts_box->value();

    HalfEdgeMesh<cVertex, cNormal> mesh(m_mesh->getMeshBuffer());

    mesh.removeDanglingArtifacts(removeDanglingArtifacts);
    mesh.finalize();

    ModelPtr model(new Model(mesh.meshBuffer()));
    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (artifacts removed)";
    m_optimizedModel = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_optimizedModel);
    m_optimizedModel->setExpanded(true);
}

}
