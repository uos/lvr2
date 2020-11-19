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
 * LVRSoilAssistBridge.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRSoilAssistBridge.hpp"
#include "LVRModelBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkPoints.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkPolygon.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkIdFilter.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkUnsignedCharArray.h>
#include "lvr2/util/Util.hpp"
#include <vtkIdTypeArray.h>

#include <vtkArrowSource.h>
#include <vtkMath.h>
#include <vtkMinimalStandardRandomSequence.h>
#include <vtkNamedColors.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

namespace lvr2
{

LVRSoilAssistBridge::LVRSoilAssistBridge(SoilAssistFieldPtr field) : m_offset_set(true)
{
    m_offset[0] = 0;
    m_offset[1] = 0;
    m_offset[2] = 0;
    if(field)
    {
        computePolygonActor(field);
    }
}

template <typename T>
bool color_equal(const color<T> &col1, const color<T> &col2)
{
    return col1.r == col2.r && col1.g == col2.g && col1.b == col2.b;
}


PolygonPtr LVRSoilAssistBridge::getPolygon()
{
    return m_polygon;
}

size_t  LVRSoilAssistBridge::getNumPoints()
{
    return 0;
}

LVRSoilAssistBridge::~LVRSoilAssistBridge()
{
}

vtkSmartPointer<vtkActor> LVRSoilAssistBridge::makeArrow(float * start, float * end)
{

    double startPoint[3];
    double endPoint[3];

    if(! m_offset_set)
    {
        m_offset[0] = start[0];
        m_offset[1] = start[1];
        m_offset[1] = start[2];
        m_offset_set = true;
    }
    startPoint[0] = start[0] - m_offset[0];
    startPoint[1] = start[1] - m_offset[1];
    startPoint[2] = start[2] - m_offset[2];
    endPoint[0] = end[0]- m_offset[0];
    endPoint[1] = end[1]- m_offset[1];
    endPoint[2] = end[2]- m_offset[2];

    std::cout << "arrow_start: \t" << startPoint[0] << "|" << startPoint[1] << "|" << startPoint[2] << std::endl;
    std::cout << "arrow_end: \t" << endPoint[0] << "|" << endPoint[1] << "|" << endPoint[2] << std::endl;


    //Create an arrow.
    vtkSmartPointer<vtkArrowSource> arrowSource =
            vtkSmartPointer<vtkArrowSource>::New();

// Compute a basis
    double normalizedX[3];
    double normalizedY[3];
    double normalizedZ[3];

    // The X axis is a vector from start to end
    vtkMath::Subtract(endPoint, startPoint, normalizedX);
    double length = vtkMath::Norm(normalizedX);
    vtkMath::Normalize(normalizedX);


    // The Z axis is an arbitrary vector cross X
    double arbitrary[3];
    for (auto i = 0; i < 3; ++i)
    {
        arbitrary[i] = 1;
    }
    vtkMath::Cross(normalizedX, arbitrary, normalizedZ);
    vtkMath::Normalize(normalizedZ);
// The Y axis is Z cross X
    vtkMath::Cross(normalizedZ, normalizedX, normalizedY);
    vtkSmartPointer<vtkMatrix4x4> matrix =
            vtkSmartPointer<vtkMatrix4x4>::New();

    // Create the direction cosine matrix
    matrix->Identity();
    for (auto i = 0; i < 3; i++)
    {
        matrix->SetElement(i, 0, normalizedX[i]);
        matrix->SetElement(i, 1, normalizedY[i]);
        matrix->SetElement(i, 2, normalizedZ[i]);
    }

    // Apply the transforms
    vtkSmartPointer<vtkTransform> transform =
            vtkSmartPointer<vtkTransform>::New();
    transform->Translate(startPoint);
    transform->Concatenate(matrix);
    transform->Scale(length, length, length);

    // Transform the polydata
    vtkSmartPointer<vtkTransformPolyDataFilter> transformPD =
            vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformPD->SetTransform(transform);
    transformPD->SetInputConnection(arrowSource->GetOutputPort());

    //Create a mapper and actor for the arrow
    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    mapper->SetInputConnection(arrowSource->GetOutputPort());
  actor->SetUserMatrix(transform->GetMatrix());

    actor->SetMapper(mapper);
    vtkSmartPointer<vtkNamedColors> colors =
            vtkSmartPointer<vtkNamedColors>::New();

    //actor->GetProperty()->SetColor(colors->GetColor3d("Cyan").GetData());
    actor->GetProperty()->SetColor(1.0, 0.0, 1.0);


    return actor;


}


vtkSmartPointer<vtkActor> LVRSoilAssistBridge::computePolygonActor(PolygonPtr poly, bool polygon)
{
    auto polygonActor = vtkSmartPointer<vtkActor>::New();

    if(poly)
    {
        floatArr points = poly->getPointArray();


        // Setup a poly data object
        vtkSmartPointer<vtkPolygon> vtk_polygon = vtkSmartPointer<vtkPolygon>::New();
        vtkSmartPointer<vtkPoints>      vtk_points = vtkSmartPointer<vtkPoints>::New();
        size_t n = poly->numPoints();


        std::cout << lvr2::timestamp << " " << n << " Points" << std::endl;
        float x0,y0,z0;
        if(m_offset_set)
        {
            x0 = m_offset[0];
            y0 = m_offset[1];
            z0 = m_offset[2];
        }
        else
        {
            x0 = points[0];
            y0 = points[1];
            z0 = points[2];
            m_offset[0] = x0;
            m_offset[1] = y0;
            m_offset[2] = z0;
            m_offset_set = true;
        }


        for(int i = 0 ; i < n ; i++)
        {
            vtk_points->InsertNextPoint( points[i*3]-x0, points[i*3+1]-y0, points[i*3+2]-z0);
            std::cout << std::fixed << "P " << points[i*3] << "|" << points[i*3+1]<< "|" << points[i*3+2] << std::endl;
        }
        //Why does this not work?
//        vtkSmartPointer<vtkFloatArray> pts_data = vtkSmartPointer<vtkFloatArray>::New();
//        pts_data->SetNumberOfComponents(3);
//        pts_data->SetVoidArray(points.get(), n * 3, 1);
//        vtk_points->SetData(pts_data);

        std::cout << lvr2::timestamp << "Mapped" << std::endl;

        //
        vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
        //todo: linestring
        lines->InsertNextCell(n+1);
        for(size_t i = 0; i < n ; i++)
        {
            lines->InsertCellPoint(i);
        }
        if(polygon)
        {
            lines->InsertCellPoint(0);
        }


//        vtk_polygon->GetPointIds()->SetNumberOfIds(n);
//        for(int i = 0 ; i < n ; i++)
//        {
//            vtk_polygon->GetPointIds()->SetId(i, i);
//        }
//
//
//        vtkSmartPointer<vtkCellArray> vtk_polygons = vtkSmartPointer<vtkCellArray>::New();
//        vtk_polygons->InsertNextCell(vtk_polygon);

        auto m_vtk_polyData = vtkSmartPointer<vtkPolyData>::New();

        m_vtk_polyData->SetPoints(vtk_points);
//        m_vtk_polyData->SetPolys(vtk_polygons);
        m_vtk_polyData->SetLines(lines);


        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

#ifdef LVR2_USE_VTK5
        mapper->SetInput(m_vtk_polyData);
#else
        mapper->SetInputData(m_vtk_polyData);
#endif
        mapper->Update();
        polygonActor->SetMapper(mapper);
        polygonActor->GetProperty()->SetColor(1.0, 0.0, 0.0);
        polygonActor->GetProperty()->SetPointSize(5);
        polygonActor->GetProperty()->SetLineWidth(4);
    }
    return polygonActor;

}

void LVRSoilAssistBridge::computePolygonActor(SoilAssistFieldPtr field)
{
    auto tmp_actor = computePolygonActor(field->getBoundary());
    tmp_actor->GetProperty()->SetColor(0.0, 0.0, 1.0);
    m_actors.push_back(tmp_actor);
    for(auto && subfield : field->getSubFields())
    {

        auto tmp_actor = computePolygonActor(subfield->getBoundary());
        tmp_actor->GetProperty()->SetColor(0.0, 0.0, 1.0);
        m_actors.push_back(tmp_actor);
        for(auto && headland : subfield->getHeadlands())
        {
            auto tmp_actor =computePolygonActor(headland);
            tmp_actor->GetProperty()->SetColor(0.0, 0.83, 1.0);
            m_actors.push_back(tmp_actor);
        }
        for(auto && line : subfield->getReferenceLines())
        {
            auto tmp_actor = computePolygonActor(line,false);
            tmp_actor->GetProperty()->SetColor(1.0, 0.0, 1.0);
            m_actors.push_back(tmp_actor);
        }
        for(auto & p : subfield->getAccessPoints())
        {
            float s[3],e[3];
            e[0] = p[0];
            e[1] = p[1];
            e[2] = p[2];

            s[0] = p[0];
            s[1] = p[1];
            s[2] = p[2]+50;
            std::cout << "ACC_: " << p[0] << "|" << p[1] << "|"<< p[2] << std::endl;
            m_actors.push_back(makeArrow(s,e));
        }

    }
}

LVRSoilAssistBridge::LVRSoilAssistBridge(const LVRSoilAssistBridge& b)
{
    m_actors   =    b.m_actors;
}

void LVRSoilAssistBridge::setBaseColor(float r, float g, float b)
{
    for(auto & actor : m_actors)
    {
        actor->GetProperty()->SetColor(r, g, b);
    }
}

void LVRSoilAssistBridge::setPointSize(int pointSize)
{

}

void LVRSoilAssistBridge::setOpacity(float opacityValue)
{
    for(auto & actor : m_actors)
    {
        vtkSmartPointer<vtkProperty> p = actor->GetProperty();
        p->SetOpacity(opacityValue);
    }

}



void LVRSoilAssistBridge::setVisibility(bool visible)
{
    for(auto & actor : m_actors)
    {
        actor->SetVisibility(visible);
    }
}



std::vector<vtkSmartPointer<vtkActor> > LVRSoilAssistBridge::getPolygonActors()
{
    return m_actors;
}


} /* namespace lvr2 */
