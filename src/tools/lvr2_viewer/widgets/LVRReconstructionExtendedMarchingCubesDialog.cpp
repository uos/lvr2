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
#include "LVRReconstructionExtendedMarchingCubesDialog.hpp"

#include "lvr2/algorithm/NormalAlgorithms.hpp"

#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/reconstruction/PointsetGrid.hpp"
#include "lvr2/reconstruction/SharpBox.hpp"

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"

#include "lvr2/io/PointBuffer.hpp"

namespace lvr2
{

LVRReconstructViaExtendedMarchingCubesDialog::LVRReconstructViaExtendedMarchingCubesDialog(string decomposition, LVRPointCloudItem* pc, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* window) :
   m_decomposition(decomposition), 
   m_pc(pc), m_parent(parent), 
   m_treeWidget(treeWidget),
   m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new ReconstructViaExtendedMarchingCubesDialog;
    m_dialog->setupUi(dialog);


    dialog->setWindowTitle("Extended Marching Cubes");

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRReconstructViaExtendedMarchingCubesDialog::~LVRReconstructViaExtendedMarchingCubesDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRReconstructViaExtendedMarchingCubesDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->comboBox_pcm, SIGNAL(currentIndexChanged(const QString)), this, SLOT(toggleRANSACcheckBox(const QString)));
    QObject::connect(m_dialog->comboBox_gs, SIGNAL(currentIndexChanged(int)), this, SLOT(switchGridSizeDetermination(int)));
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(generateMesh()));
}

void LVRReconstructViaExtendedMarchingCubesDialog::toggleRANSACcheckBox(const QString &text)
{
    QCheckBox* ransac_box = m_dialog->checkBox_RANSAC;
    if(text == "PCL")
    {
        ransac_box->setChecked(false);
        ransac_box->setCheckable(false);
    }
    else
    {
        ransac_box->setCheckable(true);
    }
}

void LVRReconstructViaExtendedMarchingCubesDialog::switchGridSizeDetermination(int index)
{
    QComboBox* gs_box = m_dialog->comboBox_gs;

    QLabel* label = m_dialog->label_below_gs;
    QDoubleSpinBox* spinBox = m_dialog->spinBox_below_gs;

    // TODO: Add reasonable default values
    if(index == 0)
    {
        label->setText("Voxel size");
        spinBox->setMinimum(0);
        spinBox->setMaximum(2000000);
        spinBox->setSingleStep(1);
        spinBox->setValue(10);
    }
    else
    {
        label->setText("Number of intersections");
        spinBox->setMinimum(1);
        spinBox->setMaximum(200000);
        spinBox->setSingleStep(1);
        spinBox->setValue(10);
    }
}

void LVRReconstructViaExtendedMarchingCubesDialog::generateMesh()
{
    using Vec = BaseVector<float>;

    QComboBox* pcm_box = m_dialog->comboBox_pcm;
    string pcm = pcm_box->currentText().toStdString();
    QCheckBox* extrusion_box = m_dialog->checkBox_Extrusion;
    bool extrusion = extrusion_box->isChecked();
    QCheckBox* ransac_box = m_dialog->checkBox_RANSAC;
    bool ransac = ransac_box->isChecked();
    QSpinBox* kn_box = m_dialog->spinBox_kn;
    int kn = kn_box->value();
    QSpinBox* kd_box = m_dialog->spinBox_kd;
    int kd = kd_box->value();
    QSpinBox* ki_box = m_dialog->spinBox_ki;
    int ki = ki_box->value();
    QCheckBox* reNormals_box = m_dialog->checkBox_renormals;
    bool reestimateNormals = reNormals_box->isChecked();
    QComboBox* gridMode_box = m_dialog->comboBox_gs;
    bool useVoxelsize = (gridMode_box->currentIndex() == 0) ? true : false;
    QDoubleSpinBox* gridSize_box = m_dialog->spinBox_below_gs;
    float  resolution = (float)gridSize_box->value();

    float  sf = m_dialog->doubleSpinBox_sf->value();
    float  sc = m_dialog->doubleSpinBox_sc->value();


    PointBufferPtr pc_buffer = m_pc->getPointBuffer();
    
    PointsetSurfacePtr<Vec> surface;

    if(pcm == "STANN" || pcm == "FLANN" || pcm == "NABO" || pcm == "NANOFLANN")
    {
        surface = PointsetSurfacePtr<Vec>( new AdaptiveKSearchSurface<Vec>(pc_buffer, pcm, kn, ki, kd, ransac ? 1 : 0) );
    }

    if(!surface->pointBuffer()->hasNormals() || reestimateNormals)
    {
        surface->calculateSurfaceNormals();
    }

    SharpBox<Vec>::m_surface     = surface;
    SharpBox<Vec>::m_theta_sharp = sf;
	SharpBox<Vec>::m_phi_corner  = sc;

    auto grid = std::make_shared<PointsetGrid<Vec, SharpBox<Vec>>>(
        resolution,
        surface,
        surface->getBoundingBox(),
        useVoxelsize,
        extrusion 
    );

    grid->calcDistanceValues();
    auto reconstruction = make_unique<FastReconstruction<Vec, SharpBox<Vec>>>(grid);

    // Create an empty mesh
    HalfEdgeMesh<Vec> mesh;
    reconstruction->getMesh(mesh);

    auto faceNormals = calcFaceNormals(mesh);

    ClusterBiMap<FaceHandle> clusterBiMap = planarClusterGrowing(mesh, faceNormals, 0.85);
    deleteSmallPlanarCluster(mesh, clusterBiMap, 10);

    ClusterPainter painter(clusterBiMap);
    auto clusterColors = DenseClusterMap<Rgb8Color>(painter.simpsons(mesh));
    auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);

    TextureFinalizer<Vec> finalize(clusterBiMap);
    finalize.setVertexNormals(vertexNormals);
    finalize.setClusterColors(clusterColors);
    Materializer<Vec> materializer(mesh, clusterBiMap, faceNormals, *surface);
    MaterializerResult<Vec> matResult = materializer.generateMaterials();
    finalize.setMaterializerResult(matResult);
    MeshBufferPtr buffer = finalize.apply(mesh);


    ModelPtr model(new Model(buffer));
	ModelBridgePtr bridge(new LVRModelBridge(model));
	
	vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
	bridge->addActors(renderer);
   
	QString base = m_parent->getName() + " (mesh)";
    m_generatedModel = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_generatedModel);
    m_generatedModel->setExpanded(true);
}

} // namespace lvr2
