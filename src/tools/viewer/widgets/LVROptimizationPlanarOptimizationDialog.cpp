#include <QFileDialog>
#include "LVROptimizationPlanarOptimizationDialog.hpp"

namespace lvr
{

LVRPlanarOptimizationDialog::LVRPlanarOptimizationDialog(LVRMeshItem* mesh, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* window) :
   m_mesh(mesh), m_parent(parent), m_treeWidget(treeWidget), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(treeWidget);
    m_dialog = new PlanarOptimizationDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRPlanarOptimizationDialog::~LVRPlanarOptimizationDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRPlanarOptimizationDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(optimizeMesh()));
    QObject::connect(m_dialog->checkBox_sr, SIGNAL(stateChanged(int)), this, SLOT(toggleSmallRegionRemoval(int)));
    QObject::connect(m_dialog->checkBox_rt, SIGNAL(stateChanged(int)), this, SLOT(toggleRetesselation(int)));
}

void LVRPlanarOptimizationDialog::toggleSmallRegionRemoval(int state)
{
    QSpinBox* smallRegionRemoval_box = m_dialog->spinBox_sr;
    if(state == Qt::CheckState::Checked)
    {
        smallRegionRemoval_box->setEnabled(true);
    }
    else
    {
        smallRegionRemoval_box->setEnabled(false);
    }
}

void LVRPlanarOptimizationDialog::toggleRetesselation(int state)
{
    QCheckBox* generateTextures_box = m_dialog->checkBox_gt;
    QDoubleSpinBox* lineSegmentThreshold_box = m_dialog->doubleSpinBox_ls;
    if(state == Qt::CheckState::Checked)
    {
        generateTextures_box->setEnabled(true);
        lineSegmentThreshold_box->setEnabled(true);
    }
    else
    {
        generateTextures_box->setEnabled(false);
        lineSegmentThreshold_box->setEnabled(false);
    }
}

void LVRPlanarOptimizationDialog::optimizeMesh()
{
    QSpinBox* planeIterations_box = m_dialog->spinBox_pi;
    int planeIterations = planeIterations_box->value();
    QDoubleSpinBox* normalThreshold_box = m_dialog->doubleSpinBox_nt;
    float normalThreshold = (float)normalThreshold_box->value();
    QSpinBox* minimalPlaneSize_box = m_dialog->spinBox_mp;
    int minimalPlaneSize = minimalPlaneSize_box->value();
    QCheckBox* removeSmallRegions_box = m_dialog->checkBox_sr;
    bool removeSmallRegions = removeSmallRegions_box->isChecked();
    QSpinBox* removeSmallRegionThreshold_box = m_dialog->spinBox_sr;
    int removeSmallRegionThreshold = (removeSmallRegions) ? removeSmallRegionThreshold_box->value() : 0;
    QCheckBox* fillHoles_box = m_dialog->checkBox_fh;
    bool fillHoles = fillHoles_box->isChecked();
    QCheckBox* retesselate_box = m_dialog->checkBox_rt;
    bool retesselate = retesselate_box->isChecked();
    QCheckBox* generateTextures_box = m_dialog->checkBox_gt;
    bool generateTextures = generateTextures_box->isChecked();
    QDoubleSpinBox* lineSegmentThreshold_box = m_dialog->doubleSpinBox_ls;
    float lineSegmentThreshold = (float)lineSegmentThreshold_box->value();

    HalfEdgeMesh<cVertex, cNormal> mesh(m_mesh->getMeshBuffer());

    mesh.optimizePlanes(planeIterations,
            normalThreshold,
            minimalPlaneSize,
            removeSmallRegionThreshold,
            true);

    mesh.fillHoles(fillHoles);
    mesh.optimizePlaneIntersections();
    mesh.restorePlanes(minimalPlaneSize);

    // Save triangle mesh
    if(retesselate)
    {
        mesh.finalizeAndRetesselate(generateTextures, lineSegmentThreshold);
    }
    else
    {
        mesh.finalize();
    }

    ModelPtr model(new Model(mesh.meshBuffer()));
    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (optimized)";
    m_optimizedModel = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_optimizedModel);
    m_optimizedModel->setExpanded(true);
}

}
