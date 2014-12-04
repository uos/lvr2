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

#include <geometry/Matrix4.hpp>
#include <geometry/Vertex.hpp>

#include <vtkTransform.h>

namespace lvr
{

LVRModelBridge::LVRModelBridge(ModelPtr model) :
    m_pointBridge(new LVRPointBufferBridge(model->m_pointCloud)),
    m_meshBridge(new LVRMeshBufferBridge(model->m_mesh))
{
    m_pose = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

LVRModelBridge::LVRModelBridge(const LVRModelBridge& b)
{
    m_pointBridge = b.m_pointBridge;
    m_meshBridge = b.m_meshBridge;
    m_pose = b.m_pose;
}

bool LVRModelBridge::validPointBridge()
{
    return (m_pointBridge->getNumPoints() > 0) ? true : false;
}

bool LVRModelBridge::validMeshBridge()
{
    return (m_meshBridge->getNumTriangles() > 0) ? true : false;
}

void LVRModelBridge::setPose(Pose& pose)
{
    m_pose = pose;
    vtkSmartPointer<vtkTransform> transform =  vtkSmartPointer<vtkTransform>::New();
    transform->PostMultiply();
    transform->RotateX(pose.r);
    transform->RotateY(pose.t);
    transform->RotateZ(pose.p);
    transform->Translate(pose.x, pose.y, pose.z);
    if(validPointBridge()) m_pointBridge->getPointCloudActor()->SetUserTransform(transform);
    if(validMeshBridge()) m_meshBridge->getMeshActor()->SetUserTransform(transform);
}

Pose LVRModelBridge::getPose()
{
    return m_pose;
}

void LVRModelBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
    if(validPointBridge())
    {
    	renderer->AddActor(m_pointBridge->getPointCloudActor());
    }

    if(validMeshBridge())
    {
    	// For simple meshes we only need to add a sigle actor to
    	// render it. For textured meshes we add a actor collection
    	// (one actor per texture).
    	if(!m_meshBridge->hasTextures())
    	{
    		renderer->AddActor(m_meshBridge->getMeshActor());
    	}
    	else
    	{
    		// TODO: Add collection to renderer
    		//renderer->AddActor(m_meshBridge->getMeshActor());
    		vtkSmartPointer<vtkActorCollection> collection = m_meshBridge->getTexturedActors();
    		collection->InitTraversal();
    		for(vtkIdType i = 0; i < collection->GetNumberOfItems(); i++)
    		{
    			cout << i << " / " << collection->GetNumberOfItems() << endl;
    			renderer->AddActor(collection->GetNextActor());
    		}

    	}
    }
}

void LVRModelBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    if(validPointBridge()) renderer->RemoveActor(m_pointBridge->getPointCloudActor());
    if(validMeshBridge()) renderer->RemoveActor(m_meshBridge->getMeshActor());
}

void LVRModelBridge::setVisibility(bool visible)
{
    if(validPointBridge()) m_pointBridge->setVisibility(visible);
    if(validMeshBridge()) m_meshBridge->setVisibility(visible);
}

LVRModelBridge::~LVRModelBridge()
{
    // TODO Auto-generated destructor stub
}

} /* namespace asteroids */
