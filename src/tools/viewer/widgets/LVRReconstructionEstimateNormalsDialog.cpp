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
    QSpinBox* spinBox_ki = m_dialog->spinBox_ki;
    int ki = spinBox_ki->value();

    PointBufferPtr pc = m_pc->getPointBuffer();
    size_t numPoints = m_pc->getNumPoints();
    floatArr points = pc->getPointArray(numPoints);
    floatArr normals = pc->getPointNormalArray(numPoints);

    typename SearchTree<Vertex<float> >::Ptr       tree;
    #ifdef _USE_PCL_
            tree = SearchTree<Vertex<float> >::Ptr( new SearchTreeFlann<Vertex<float> >(pc, numPoints, ki, ki, ki) );
    #else
            cout << timestamp << "Warning: PCL is not installed. Using STANN search tree in AdaptiveKSearchSurface." << endl;
            tree = SearchTree<Vertex<float> >::Ptr( new SearchTreeStann<Vertex<float> >(pc, numPoints, ki, ki, ki) );
    #endif

    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < numPoints; i++)
    {
        // Create search tree
        vector< ulong > indices;
        vector< double > distances;

        Vertex<float> vertex(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
        tree->kSearch(vertex, ki, indices, distances);

        // Do interpolation
        Normal<float> normal(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]);
        for(int j = 0; j < indices.size(); j++)
        {
            normal += Normal<float>(normals[3 * indices[j]], normals[3 * indices[j] + 1], normals[3 * indices[j] + 2]);
        }
        normal.normalize();

        // Save results in buffer (I know i should use a seperate buffer, but for testing it
        // should be OK to save the interpolated values directly into the input buffer)
        normals[3 * i]      = normal.x;
        normals[3 * i + 1]  = normal.y;
        normals[3 * i + 2]  = normal.z;
    }

    ModelPtr model(new Model(pc));

    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (w. normals)";
    m_pointCloudWithNormals = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_pointCloudWithNormals);
    m_pointCloudWithNormals->setExpanded(true);
}

}
