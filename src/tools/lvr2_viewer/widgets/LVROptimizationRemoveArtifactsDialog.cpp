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

#include "LVROptimizationRemoveArtifactsDialog.hpp"

#include "lvr2/algorithm/ClusterAlgorithms.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/io/Model.hpp"

namespace lvr2
{

LVRRemoveArtifactsDialog::LVRRemoveArtifactsDialog(LVRMeshItem* mesh, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* window) :
   m_mesh(mesh), m_parent(parent), m_treeWidget(treeWidget), m_renderWindow(window)
{
    // Setup DialogUI and events
    QDialog* dialog = new QDialog(m_treeWidget);
    m_dialog = new RemoveArtifactsDialog;
    m_dialog->setupUi(dialog);

    connectSignalsAndSlots();

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

LVRRemoveArtifactsDialog::~LVRRemoveArtifactsDialog()
{
    // TODO Auto-generated destructor stub
}

void LVRRemoveArtifactsDialog::connectSignalsAndSlots()
{
    QObject::connect(m_dialog->buttonBox, SIGNAL(accepted()), this, SLOT(removeArtifacts()));
}

void LVRRemoveArtifactsDialog::removeArtifacts()
{
    using Vec = BaseVector<float>;

    QSpinBox* removeDanglingArtifacts_box = m_dialog->spinBox_rda;
    int removeDanglingArtifacts = removeDanglingArtifacts_box->value();

    HalfEdgeMesh<Vec> mesh(m_mesh->getMeshBuffer());
    removeDanglingCluster(mesh, removeDanglingArtifacts);

    // create normals and/or colors?
    SimpleFinalizer<Vec> fin;

    ModelPtr model(new Model(fin.apply(mesh)));
    ModelBridgePtr bridge(new LVRModelBridge(model));
    vtkSmartPointer<vtkRenderer> renderer = m_renderWindow->GetRenderers()->GetFirstRenderer();
    bridge->addActors(renderer);

    QString base = m_parent->getName() + " (artifacts removed)";
    m_optimizedModel = new LVRModelItem(bridge, base);

    m_treeWidget->addTopLevelItem(m_optimizedModel);
    m_optimizedModel->setExpanded(true);
}

} // namespace lvr2
