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

    PointBufferPtr pc = m_pc->getPointBuffer();
    floatArr old_pts = pc->getPointArray();
    size_t numPoints = m_pc->getNumPoints();

    // Create buffer arrays
    floatArr points(new float[3 * numPoints]);

    // copy pts to new pointbuffer 
    std::copy(old_pts.get(), old_pts.get() + numPoints*3, points.get());

    PointBufferPtr new_pc = PointBufferPtr( new PointBuffer );
    new_pc->setPointArray(points, numPoints);

    // with k == 0 no normal interpolation
    int k = interpolateNormals ? 10 : 0;

    AdaptiveKSearchSurface<Vec> surface(new_pc, "FLANN", ki, k, k);
    surface.calculateSurfaceNormals();

    ModelPtr model(new Model(new_pc));

    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (w. normals)";
    m_pointCloudWithNormals = new LVRModelItem(bridge, base);
    m_pointCloudWithNormals->setPose(m_parent->getPose());

    m_treeWidget->addTopLevelItem(m_pointCloudWithNormals);
    m_pointCloudWithNormals->setExpanded(true);
}

} // namespace lvr2
