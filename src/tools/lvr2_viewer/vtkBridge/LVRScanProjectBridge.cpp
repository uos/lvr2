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

/**
 * LVRModel.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRScanProjectItem.hpp"
#include "LVRModelBridge.hpp"

#include "lvr2/geometry/Matrix4.hpp"

#include <vtkTransform.h>
#include <vtkActor.h>
#include <vtkProperty.h>

namespace lvr2
{

class LVRMeshBufferBridge;

ScanProjectBridge::ScanProjectBridge(ScanProjectPtr project) :
{
    for (auto position : model->positions)
    {
        for(auto scan : model->scans)
        {
            ModelPtr model(new Model);
            PointBufferPtr points(new PointBufferPtr(scan->points));
            model->m_pointCloud = points;
               //TODO set pose?
            models.push_back(model);
        }        
    }
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

void LVRModelBridge::setPose(const Pose& pose)
{
    m_pose = pose;
    vtkSmartPointer<vtkTransform> transform =  vtkSmartPointer<vtkTransform>::New();
    transform->PostMultiply();
    transform->RotateX(pose.r);
    transform->RotateY(pose.t);
    transform->RotateZ(pose.p);
    transform->Translate(pose.x, pose.y, pose.z);

    doStuff(transform);

}

void LVRModelBridge::setTransform(const Transformd &transform)
{
    vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
    vtkSmartPointer<vtkMatrix4x4> m = vtkSmartPointer<vtkMatrix4x4>::New();

    // For some reason we have to copy the matrix
    // values manually...
    const double* arr = transform.data();
    int j = 0;
    for(int i = 0; i < 16; i++)
    {
        if((i % 4) == 0)
        {
            j = 0;
        }
        double v = arr[i];
        m->SetElement(i / 4, j, v);
        j++;
    }

    t->PostMultiply();
    t->SetMatrix(m);
    doStuff(t);
}

void LVRModelBridge::doStuff(vtkSmartPointer<vtkTransform> transform)
{
    if(validPointBridge())
    {
        m_pointBridge->getPointCloudActor()->SetUserTransform(transform);
    }

    if(validMeshBridge())
    {
        if (!m_meshBridge->hasTextures())
        {
            m_meshBridge->getMeshActor()->SetUserTransform(transform);
        }
        else
        {
            vtkSmartPointer<vtkActorCollection> col = m_meshBridge->getTexturedActors();
            col->InitTraversal();
            for (int i = 0; i < col->GetNumberOfItems(); i++)
            {
                col->GetNextActor()->SetUserTransform(transform);
            }
        }
    }
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
}

void LVRModelBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    if(validPointBridge()) renderer->RemoveActor(m_pointBridge->getPointCloudActor());
}


} /* namespace lvr2 */
