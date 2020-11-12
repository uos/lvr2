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
 * LVRPolygonBridge.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRPolygonBridge.hpp"
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
namespace lvr2
{

LVRPolygonBridge::LVRPolygonBridge(PolygonPtr poly)
{

    m_vtk_polyData = vtkSmartPointer<vtkPolyData>::New();

    m_numPoints = 0;

    if(poly)
    {
//        // Save pc data
//        m_pointBuffer = pointCloud;
//
//        if(pointCloud->hasColors()) m_hasColors = true;
//        if(pointCloud->hasNormals()) m_hasNormals = true;
//
//        // default: visible light
//        m_spectralChannels.r = Util::getSpectralChannel(612, pointCloud);
//        m_spectralChannels.g = Util::getSpectralChannel(552, pointCloud);
//        m_spectralChannels.b = Util::getSpectralChannel(462, pointCloud);
//
//        // Generate vtk actor representation
//        computePointCloudActor(pointCloud);
//
//        // Save meta information
//        m_numPoints = pointCloud->numPoints();
    m_numPoints = poly->numPoints();
    std::cout << "A num " << m_numPoints << std::endl;
    computePolygonActor(poly);
    }
}

template <typename T>
bool color_equal(const color<T> &col1, const color<T> &col2)
{
    return col1.r == col2.r && col1.g == col2.g && col1.b == col2.b;
}


PolygonPtr LVRPolygonBridge::getPolygon()
{
    return m_polygon;
}

size_t  LVRPolygonBridge::getNumPoints()
{
    return m_numPoints;
}

LVRPolygonBridge::~LVRPolygonBridge()
{
}

void LVRPolygonBridge::computePolygonActor(PolygonPtr poly)
{
    if(poly)
    {
        floatArr points = poly->getPointArray();

        m_PolygonActor = vtkSmartPointer<vtkActor>::New();

        // Setup a poly data object
        vtkSmartPointer<vtkPolygon> vtk_polygon = vtkSmartPointer<vtkPolygon>::New();
        vtkSmartPointer<vtkPoints>      vtk_points = vtkSmartPointer<vtkPoints>::New();
        size_t n = poly->numPoints();


        std::cout << lvr2::timestamp << " " << n << " Points" << std::endl;
        for(int i = 0 ; i < n ; i++)
        {
            vtk_points->InsertNextPoint( points[i*3], points[i*3+1], points[i*3+2]);
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
        lines->InsertCellPoint(0);


//        vtk_polygon->GetPointIds()->SetNumberOfIds(n);
//        for(int i = 0 ; i < n ; i++)
//        {
//            vtk_polygon->GetPointIds()->SetId(i, i);
//        }
//
//
//        vtkSmartPointer<vtkCellArray> vtk_polygons = vtkSmartPointer<vtkCellArray>::New();
//        vtk_polygons->InsertNextCell(vtk_polygon);

        m_vtk_polyData = vtkSmartPointer<vtkPolyData>::New();

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
        m_PolygonActor->SetMapper(mapper);
        m_PolygonActor->GetProperty()->SetColor(1.0, 0.0, 0.0);
        m_PolygonActor->GetProperty()->SetPointSize(5);
        m_PolygonActor->GetProperty()->SetLineWidth(4);
    }
}

LVRPolygonBridge::LVRPolygonBridge(const LVRPolygonBridge& b)
{
    m_PolygonActor   =    b.m_PolygonActor;
    m_numPoints         = b.m_numPoints;
    m_polygon           =b.m_polygon;
}

void LVRPolygonBridge::setBaseColor(float r, float g, float b)
{
    m_PolygonActor->GetProperty()->SetColor(r, g, b);
}

void LVRPolygonBridge::setPointSize(int pointSize)
{
    vtkSmartPointer<vtkProperty> p = m_PolygonActor->GetProperty();
    p->SetPointSize(pointSize);
}

void LVRPolygonBridge::setOpacity(float opacityValue)
{
    vtkSmartPointer<vtkProperty> p = m_PolygonActor->GetProperty();
    p->SetOpacity(opacityValue);
}



void LVRPolygonBridge::setVisibility(bool visible)
{
    m_PolygonActor->SetVisibility(visible);
}



vtkSmartPointer<vtkActor> LVRPolygonBridge::getPolygonActor()
{
    return m_PolygonActor;
}


} /* namespace lvr2 */
