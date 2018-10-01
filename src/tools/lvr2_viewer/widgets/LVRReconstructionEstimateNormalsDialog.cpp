#include <QFileDialog>
#include "LVRReconstructionEstimateNormalsDialog.hpp"

#include <lvr2/geometry/BaseVector.hpp>

#include <lvr2/io/DataStruct.hpp>

#include <lvr2/reconstruction/AdaptiveKSearchSurface.hpp>

namespace lvr2
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
    if(state == Qt::Checked)
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
    using Vec = BaseVector<float>;

    QCheckBox* checkBox_in = m_dialog->checkBox_in;
    bool interpolateNormals = checkBox_in->isChecked();
    QSpinBox* spinBox_ki = m_dialog->spinBox_ki;
    int ki = spinBox_ki->value();

    PointBuffer2Ptr pc = m_pc->getPointBuffer();
    floatArr old_pts = pc->getPointArray();
    size_t numPoints = m_pc->getNumPoints();

    // Create buffer arrays
    floatArr points(new float[3 * numPoints]);

    // old code tried to transform the point, I don't see a reason why 
    // Get transformation from frames or pose files if possible
    /*
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
    */

    // copy pts for new cp
    for (size_t i = 0; i < numPoints*3; i++)
    {
        points[i] = old_pts[i]; 
    }

    PointBuffer2Ptr new_pc = PointBuffer2Ptr( new PointBuffer2 );
    new_pc->setPointArray(points, numPoints);

    int k = 0;
    if (interpolateNormals)
    {
        k = 10;     
    }

    AdaptiveKSearchSurface<Vec> surface(new_pc, "FLANN", ki, k, k);
    surface.calculateSurfaceNormals();

    ModelPtr model(new Model(new_pc));

    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (w. normals)";
    m_pointCloudWithNormals = new LVRModelItem(bridge, base);
    //m_pointCloudWithNormals->setPose(m_parent->getPose());

    m_treeWidget->addTopLevelItem(m_pointCloudWithNormals);
    m_pointCloudWithNormals->setExpanded(true);
}

} // namespace lvr2
