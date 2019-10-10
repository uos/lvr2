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
#include "LVRItemTypes.hpp"

#include "lvr2/geometry/BaseVector.hpp"

#include "lvr2/io/DataStruct.hpp"

#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"

#if defined CUDA_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/cuda/CudaSurface.hpp"

    typedef lvr2::CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/opencl/ClSurface.hpp"
    typedef lvr2::ClSurface GpuSurface;
#endif

namespace lvr2
{

// old
LVREstimateNormalsDialog::LVREstimateNormalsDialog(LVRPointCloudItem* pc, QTreeWidgetItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* window) :
   m_pc(pc), m_parent(parent), m_treeWidget(treeWidget), m_renderWindow(window)
{
    m_pcs.append(pc);
    m_parents.append(parent);
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new EstimateNormalsDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();

    // init defaults
    toggleSpecialOptions(m_dialog->comboBox_algo_select->currentText());
}

// new
LVREstimateNormalsDialog::LVREstimateNormalsDialog(
        QList<LVRPointCloudItem*> pc_items,
        QList<QTreeWidgetItem*> parents,
        QTreeWidget* treeWidget,
        vtkRenderWindow* renderer
    )
:m_pcs(pc_items)
,m_parents(parents)
,m_treeWidget(treeWidget)
,m_renderWindow(renderer)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new EstimateNormalsDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();

    // init defaults
    toggleSpecialOptions(m_dialog->comboBox_algo_select->currentText());
}

LVREstimateNormalsDialog::~LVREstimateNormalsDialog()
{
    // TODO Auto-generated destructor stub
}

void LVREstimateNormalsDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(estimateNormals()));
    QObject::connect(m_dialog->checkBox_kn_auto, SIGNAL(stateChanged(int)), this, SLOT(toggleAutoNormalEstimation(int)));
    QObject::connect(m_dialog->checkBox_ki_auto, SIGNAL(stateChanged(int)), this, SLOT(toggleAutoNormalInterpolation(int)));
    QObject::connect(m_dialog->comboBox_algo_select, SIGNAL(currentTextChanged(QString)), this, SLOT(toggleSpecialOptions(QString)));
}

void LVREstimateNormalsDialog::toggleAutoNormalEstimation(int state)
{
    QSpinBox* spinBox_kn = m_dialog->spinBox_kn;
    if(state == Qt::Checked)
    {
        spinBox_kn->setEnabled(false);
    }
    else
    {
        spinBox_kn->setEnabled(true);
    }
}

void LVREstimateNormalsDialog::toggleAutoNormalInterpolation(int state)
{
    QSpinBox* spinBox_ki = m_dialog->spinBox_ki;
    if(state == Qt::Checked)
    {
        spinBox_ki->setEnabled(false);
    }
    else
    {
        spinBox_ki->setEnabled(true);
    }
}

void LVREstimateNormalsDialog::toggleSpecialOptions(QString current_text)
{
    std::string alog_str = current_text.toStdString();
    if(alog_str == "GPU")
    {
        m_dialog->label_fp->setVisible(true);
        m_dialog->doubleSpinBox_fp_x->setVisible(true);
        m_dialog->doubleSpinBox_fp_y->setVisible(true);
        m_dialog->doubleSpinBox_fp_z->setVisible(true);
    } else {
        m_dialog->label_fp->setVisible(false);
        m_dialog->doubleSpinBox_fp_x->setVisible(false);
        m_dialog->doubleSpinBox_fp_y->setVisible(false);
        m_dialog->doubleSpinBox_fp_z->setVisible(false);
    }
    
}

void LVREstimateNormalsDialog::estimateNormals()
{
    using Vec = BaseVector<float>;

    bool autoKn = m_dialog->checkBox_kn_auto->isChecked();
    bool autoKi = m_dialog->checkBox_ki_auto->isChecked();
    bool genNew = m_dialog->checkBox_new_item->isChecked();

    int kn = m_dialog->spinBox_kn->value();
    int ki = m_dialog->spinBox_ki->value();

    QString algo = m_dialog->comboBox_algo_select->currentText();
    std::string algo_str = algo.toStdString();

    // print stuff
    std::cout << "NORMAL ESTIMATION SETTINGS: " << std::endl;
    if(autoKn)
    {
        std::cout << "-- kn: auto" << std::endl;
    } else {
        std::cout << "-- kn: " << kn << std::endl;
    }

    if(autoKi)
    {
        std::cout << "-- ki: auto" << std::endl;
    } else {
        std::cout << "-- ki: " << ki << std::endl;
    }

    std::cout << "-- algo: " << algo_str << std::endl;

    for(int pc_id = 0; pc_id < m_pcs.size(); pc_id++)
    {
        LVRPointCloudItem* pc_item = m_pcs[pc_id];
        QTreeWidgetItem* parent = m_parents[pc_id];
        // create new point cloud
        PointBufferPtr pc = pc_item->getPointBuffer();
        size_t numPoints = pc_item->getNumPoints();

        if(genNew)
        {
            floatArr old_pts = pc->getPointArray();
            // Create buffer arrays
            floatArr points(new float[3 * numPoints]);

            // copy pts to new pointbuffer 
            std::copy(old_pts.get(), old_pts.get() + numPoints*3, points.get());

            pc.reset(new PointBuffer);
            pc->setPointArray(points, numPoints);
        }
        
        PointsetSurfacePtr<Vec> surface;

        if(algo_str == "STANN" || algo_str == "FLANN" || algo_str == "NABO" || algo_str == "NANOFLANN")
        {
            surface = std::make_shared<AdaptiveKSearchSurface<Vec> >(
                pc, algo_str, kn, ki, 20, false
            );
        } else if(algo_str == "GPU") {
            surface = std::make_shared<AdaptiveKSearchSurface<Vec> >(
                pc, "FLANN", kn, ki, 20, false
            );
        }

        if(autoKn || autoKi)
        {
            const BoundingBox<Vec>& bb = surface->getBoundingBox();
            double V = bb.getXSize() * bb.getYSize() * bb.getZSize();

            if(autoKn)
            {
                kn = static_cast<int>(
                    sqrt(static_cast<double>(numPoints)) / V * 400.0);
                std::cout << "-- auto kn: " << kn << std::endl;
            }

            if(autoKi)
            {
                ki = static_cast<int>(
                    sqrt(static_cast<double>(numPoints)) / V * 400.0);

                std::cout << "-- auto ki: " << ki << std::endl;
            }

        }
        
        if(algo_str == "GPU")
        {
            #ifdef GPU_FOUND
            // TODO
            float fpx = static_cast<float>(m_dialog->doubleSpinBox_fp_x->value());
            float fpy = static_cast<float>(m_dialog->doubleSpinBox_fp_y->value());
            float fpz = static_cast<float>(m_dialog->doubleSpinBox_fp_z->value());

            std::vector<float> flipPoint = {fpx, fpy, fpz};
            size_t num_points = pc->numPoints();
            floatArr points = pc->getPointArray();
            floatArr normals = floatArr(new float[ num_points * 3 ]);
            std::cout << timestamp << "Generate GPU kd-tree..." << std::endl;
            GpuSurface gpu_surface(points, num_points);

            gpu_surface.setKn(kn);
            gpu_surface.setKi(ki);
            gpu_surface.setFlippoint(flipPoint[0], flipPoint[1], flipPoint[2]);

            std::cout << timestamp << "Calculated normals..." << std::endl;
            gpu_surface.calculateNormals();
            gpu_surface.getNormals(normals);

            pc->setNormalArray(normals, num_points);
            gpu_surface.freeGPU();
            
            #else
            std::cout << "ERROR: GPU Driver not installed, using FLANN instead" << std::endl;
            surface->calculateSurfaceNormals();
            #endif
        } else {
            surface->calculateSurfaceNormals();
        }

        if(genNew)
        {
            ModelPtr model(new Model(pc));

            ModelBridgePtr bridge(new LVRModelBridge(model));
            vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
            bridge->addActors(renderer);

            QString base;
            if (parent->type() == LVRModelItemType)
            {
                LVRModelItem *model_item = static_cast<LVRModelItem *>(parent);
                base = model_item->getName() + " (w. normals)";
                m_pointCloudWithNormals = new LVRModelItem(bridge, base);
                m_pointCloudWithNormals->setPose(model_item->getPose());
            }
            else if (parent->type() == LVRScanDataItemType)
            {
                LVRScanDataItem *sd_item = static_cast<LVRScanDataItem *>(parent);
                base = sd_item->getName() + " (w. normals)";
                m_pointCloudWithNormals = new LVRModelItem(bridge, base);
                m_pointCloudWithNormals->setPose(sd_item->getPose());
            }

            m_treeWidget->addTopLevelItem(m_pointCloudWithNormals);
            m_pointCloudWithNormals->setExpanded(true);
        
        } else {

            if(parent->type() == LVRScanDataItemType)
            {
                // pc_item->update();
                LVRScanDataItem *sd_item = static_cast<LVRScanDataItem *>(parent);
                sd_item->reload();
            } else {
                vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
                renderer->RemoveActor(pc_item->getPointBufferBridge()->getPointCloudActor());
                pc_item->update();
                renderer->AddActor(pc_item->getPointBufferBridge()->getPointCloudActor());
            }
            
        }

        m_renderWindow->Render();

        std::cout << timestamp << "Finished." << std::endl;
        
        // m_renderWindow->GetRenderers()->GetFirstRenderer()->render();

    }
}

} // namespace lvr2
