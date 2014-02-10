/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/**
 * MainWindow.cpp
 *
 *  @date Jan 31, 2014
 *  @author Thomas Wiemann
 */

#include <QFileInfo>

#include "LVRMainWindow.hpp"
#include "../vtkBridge/LVRModelBridge.hpp"
#include "../widgets/LVRModelItem.hpp"

#include "io/ModelFactory.hpp"

namespace lvr
{

LVRMainWindow::LVRMainWindow()
{
    setupUi(this);
    setupQVTK();
    connectSignalsAndSlots();

    QHeaderView* v = this->treeWidget->header();
    v->resizeSection(0, 150);
}



LVRMainWindow::~LVRMainWindow()
{
    // TODO Auto-generated destructor stub
}

void LVRMainWindow::setupQVTK()
{
    // Add new renderer to the render window of the QVTKWidget
    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    this->qvtkWidget->GetRenderWindow()->AddRenderer(m_renderer);
}

void LVRMainWindow::connectSignalsAndSlots()
{
    QObject::connect(actionOpen, SIGNAL(activated()), this, SLOT(loadModel()));
}

void LVRMainWindow::loadModel()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Model"), "", tr("Model Files (*.ply *.obj *.pts *.3d *.txt)"));

    // Load model and generate vtk representation
    ModelPtr model = ModelFactory::readModel(filename.toStdString());
    ModelBridgePtr bridge( new LVRModelBridge(model));
    bridge->addActors(m_renderer);

    // Add item for this model to tree widget
    QFileInfo info(filename);
    QString base = info.fileName();
    LVRModelItem* item = new LVRModelItem(bridge, base);
    this->treeWidget->addTopLevelItem(item);
    item->setExpanded(true);

    // Update camera to new scene dimension and force rendering
    m_renderer->ResetCamera();
    this->qvtkWidget->GetRenderWindow()->Render();

}

} /* namespace lvr */
