#include <QFileDialog>
#include "LVROptimizationRemoveArtifactsDialog.hpp"

#include <lvr2/algorithm/ClusterAlgorithms.hpp>

#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>

#include <lvr2/io/Model.hpp>

namespace lvr2
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
    using Vec = BaseVector<float>;

    QSpinBox* removeDanglingArtifacts_box = m_dialog->spinBox_rda;
    int removeDanglingArtifacts = removeDanglingArtifacts_box->value();

    HalfEdgeMesh<Vec> mesh(m_mesh->getMeshBuffer());
    removeDanglingCluster(mesh, removeDanglingArtifacts);

    // create normals and/or colors?
    SimpleFinalizer<Vec> fin;

    ModelPtr model(new Model(fin.apply(mesh)));
    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (artifacts removed)";
    m_optimizedModel = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_optimizedModel);
    m_optimizedModel->setExpanded(true);
}

} // namespace lvr2
