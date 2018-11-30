/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
