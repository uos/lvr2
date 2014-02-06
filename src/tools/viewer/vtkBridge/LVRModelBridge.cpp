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
 * LVRModel.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRModelBridge.hpp"

namespace lvr
{

LVRModelBridge::LVRModelBridge(ModelPtr model)
    : m_pointBridge(model->m_pointCloud), m_meshBridge(model->m_mesh)
{

}

void LVRModelBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
    renderer->AddActor(m_pointBridge.getPointCloudActor());
    renderer->AddActor(m_meshBridge.getMeshActor());
}

void LVRModelBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    renderer->RemoveActor(m_pointBridge.getPointCloudActor());
    renderer->RemoveActor(m_meshBridge.getMeshActor());
}

LVRModelBridge::~LVRModelBridge()
{
    // TODO Auto-generated destructor stub
}

} /* namespace asteroids */
