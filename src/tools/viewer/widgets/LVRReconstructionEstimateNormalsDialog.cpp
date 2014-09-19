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
    QObject::connect(m_dialog->checkBox_in, SIGNAL(stateChanged(int)), this, SLOT(toggleNormalInterpolation(int)));
}

void LVREstimateNormalsDialog::toggleNormalInterpolation(int state)
{
    QSpinBox* spinBox_ki = m_dialog->spinBox_ki;
    if(state == Qt::CheckState::Checked)
    {
        spinBox_ki->setEnabled(true);
    }
    else
    {
        spinBox_ki->setEnabled(false);
    }
}

void LVREstimateNormalsDialog::estimateNormals()
{
    QCheckBox* checkBox_in = m_dialog->checkBox_in;
    bool interpolateNormals = checkBox_in->isChecked();
    QSpinBox* spinBox_ki = m_dialog->spinBox_ki;
    int ki = spinBox_ki->value();

    PointBufferPtr pc = m_pc->getPointBuffer();
    size_t numPoints = m_pc->getNumPoints();

    // Create buffer arrays
    floatArr points(new float[3 * numPoints]);
    floatArr normals(new float[3 * numPoints]);

    // Get transformation from frames or pose files if possible
    Matrix4<float> transform;
    // Matrix4 does not support lvr::Pose, convert to float-Array
    // TODO: fix transformation
    Pose pose = m_parent->getPose();
    float* float_pose = new float[6];
    float_pose[0] = pose.x;
    float_pose[1] = pose.y;
    float_pose[2] = pose.z;
    float_pose[3] = pose.r;
    float_pose[4] = pose.t;
    float_pose[5] = pose.p;
    transform.toPostionAngle(float_pose);

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

    if(interpolateNormals)
    {
        typename SearchTree<Vertex<float> >::Ptr       tree;
        #ifdef _USE_PCL_
            tree = SearchTree<Vertex<float> >::Ptr( new SearchTreeFlann<Vertex<float> >(new_pc, numPoints, ki, ki, ki) );
        #else
            cout << timestamp << "Warning: PCL is not installed. Using STANN search tree in AdaptiveKSearchSurface." << endl;
            tree = SearchTree<Vertex<float> >::Ptr( new SearchTreeStann<Vertex<float> >(new_pc, numPoints, ki, ki, ki) );
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
    }

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
