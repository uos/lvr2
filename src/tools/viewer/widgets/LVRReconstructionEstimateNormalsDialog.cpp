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

    // Create buffer arrays
    floatArr points(new float[3 * numPoints]);
    floatArr normals(new float[3 * numPoints]);

    // Get transformation from frames or pose files if possible
    Matrix4<float> transform;
    //transform.toPostionAngle(m_parent->getPose());
    // Matrix4 unterstützt lvr::Pose noch nicht

    float x, y, z, nx, ny, nz;
    size_t pointsRead = 0;

    do
    {
        // Transform normal according to pose
        Normal<float> normal(nx, ny, nz);
        Vertex<float> point(x, y, z);
        normal = transform * normal;
        point = transform * point;

        // Write data into buffer
        points[pointsRead * 3]     = point.x;
        points[pointsRead * 3 + 1] = point.y;
        points[pointsRead * 3 + 2] = point.z;

        normals[pointsRead * 3]     = normal.x;
        normals[pointsRead * 3 + 1] = normal.y;
        normals[pointsRead * 3 + 2] = normal.z;
        pointsRead++;
    } while(pointsRead < numPoints);

    PointBufferPtr new_pc = PointBufferPtr( new PointBuffer );
    new_pc->setPointArray(points, numPoints);
    new_pc->setPointNormalArray(normals, numPoints);

    ModelPtr model(new Model(new_pc));

    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (w. normals)";
    m_pointCloudWithNormals = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_pointCloudWithNormals);
    m_pointCloudWithNormals->setExpanded(true);
}

}
