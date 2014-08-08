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
 * LVRMeshBufferBridge.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRMeshBufferBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkPoints.h>
#include <vtkActor.h>
#include <vtkTriangle.h>
#include <vtkProperty.h>

namespace lvr
{

LVRMeshBufferBridge::LVRMeshBufferBridge(MeshBufferPtr meshBuffer) :
        m_meshBuffer(meshBuffer)
{
    if(meshBuffer)
    {
        computeMeshActor(meshBuffer);
        meshBuffer->getVertexArray(m_numVertices);
        meshBuffer->getFaceArray(m_numFaces);
    }
    else
    {
        m_numFaces = 0;
        m_numVertices = 0;
    }
}

void LVRMeshBufferBridge::setBaseColor(float r, float g, float b)
{
	vtkSmartPointer<vtkProperty> p = m_meshActor->GetProperty();
    p->SetColor(r, g, b);
    m_meshActor->SetProperty(p);
}

LVRMeshBufferBridge::LVRMeshBufferBridge(const LVRMeshBufferBridge& b)
{
    m_numVertices   = b.m_numVertices;
    m_numFaces      = b.m_numFaces;
    m_meshActor     = b.m_meshActor;
}

size_t LVRMeshBufferBridge::getNumTriangles()
{
    return m_numFaces;
}

size_t LVRMeshBufferBridge::getNumVertices()
{
    return m_numVertices;
}

MeshBufferPtr  LVRMeshBufferBridge::getMeshBuffer()
{
    return m_meshBuffer;
}

LVRMeshBufferBridge::~LVRMeshBufferBridge()
{
    // TODO Auto-generated destructor stub
}

void LVRMeshBufferBridge::computeMeshActor(MeshBufferPtr meshbuffer)
{
    if(meshbuffer)
    {
        vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();

        // Parse vertex and index buffer
        size_t n_v, n_i;
        floatArr vertices = meshbuffer->getVertexArray(n_v);
        uintArr indices = meshbuffer->getFaceArray(n_i);

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();

        for(size_t i = 0; i < n_v; i++){
            size_t index = 3 * i;
            points->InsertNextPoint(
                    vertices[index    ],
                    vertices[index + 1],
                    vertices[index + 2]);
        }

        for(size_t i = 0; i < n_i; i++)
        {
            size_t index = 3 * i;
            vtkSmartPointer<vtkTriangle> t = vtkSmartPointer<vtkTriangle>::New();
            t->GetPointIds()->SetId(0, indices[index]);
            t->GetPointIds()->SetId(1, indices[index + 1]);
            t->GetPointIds()->SetId(2, indices[index + 2]);
            triangles->InsertNextCell(t);
        }

        mesh->SetPoints(points);
        mesh->SetPolys(triangles);

        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInput(mesh);


        m_meshActor = vtkSmartPointer<vtkActor>::New();
        m_meshActor->SetMapper(mapper);
    }
}

void LVRMeshBufferBridge::setOpacity(float opacityValue)
{
	vtkSmartPointer<vtkProperty> p = m_meshActor->GetProperty();
    p->SetOpacity(opacityValue);
    m_meshActor->SetProperty(p);
}

void LVRMeshBufferBridge::setVisibility(bool visible)
{
    if(visible) m_meshActor->VisibilityOn();
    else m_meshActor->VisibilityOff();
}

void LVRMeshBufferBridge::setShading(int shader)
{
    vtkSmartPointer<vtkProperty> p = m_meshActor->GetProperty();
    p->SetShading(shader);
    m_meshActor->SetProperty(p);
}

vtkSmartPointer<vtkActor> LVRMeshBufferBridge::getMeshActor()
{
    return m_meshActor;
}

} /* namespace lvr */
