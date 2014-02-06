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
#include "LVRMainWindow.hpp"
#include "../vtkBridge/LVRModelBridge.hpp"

#include "io/ModelFactory.hpp"

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>

namespace lvr
{

LVRMainWindow::LVRMainWindow()
{
    setupUi(this);
    connectSignalsAndSlots();
}



LVRMainWindow::~LVRMainWindow()
{
    // TODO Auto-generated destructor stub
}

void LVRMainWindow::connectSignalsAndSlots()
{
    QObject::connect(actionOpen, SIGNAL(activated()), this, SLOT(loadModel()));
}

void LVRMainWindow::loadModel()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Model"), "", tr("Model Files (*.ply *.obj *.pts *.3d *.txt)"));
    ModelPtr model = ModelFactory::readModel(filename.toStdString());
    LVRModelBridge bridge(model);
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    this->qvtkWidget->GetRenderWindow()->AddRenderer(renderer);
    bridge.addActors(renderer);
}

} /* namespace lvr */
