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
}

void LVRPlanarOptimizationDialog::optimizeMesh()
{
    HalfEdgeMesh<cVertex, cNormal> mesh(m_mesh->getMeshBuffer());

    /*if(options.optimizePlanes())
    {
        mesh.optimizePlanes(options.getPlaneIterations(),
                options.getNormalThreshold(),
                options.getMinPlaneSize(),
                options.getSmallRegionThreshold(),
                true);

        mesh.fillHoles(options.getFillHoles());
        mesh.optimizePlaneIntersections();
        mesh.restorePlanes(options.getMinPlaneSize());

        if(options.getNumEdgeCollapses())
        {
            QuadricVertexCosts<cVertex, cNormal> c = QuadricVertexCosts<cVertex, cNormal>(true);
            mesh.reduceMeshByCollapse(options.getNumEdgeCollapses(), c);
        }
    }
    else if(options.clusterPlanes())
    {
        mesh.clusterRegions(options.getNormalThreshold(), options.getMinPlaneSize());
        mesh.fillHoles(options.getFillHoles());
    }

    // Save triangle mesh
    if ( options.retesselate() )
    {
        mesh.finalizeAndRetesselate(options.generateTextures(), options.getLineFusionThreshold());
    }
    else
    {
        mesh.finalize();
    }*/
}

}
