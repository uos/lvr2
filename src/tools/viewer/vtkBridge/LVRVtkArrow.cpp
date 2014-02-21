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
 * LVRVtkArrow.cpp
 *
 *  @date Feb 21, 2014
 *  @author Thomas Wiemann
 */
#include "LVRVtkArrow.hpp"

#include "geometry/Normal.hpp"

#include <vtkArrowSource.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkMath.h>
#include <vtkSphereSource.h>
#include <vtkProperty.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

namespace lvr
{



LVRVtkArrow::LVRVtkArrow(Vertexf start, Vertexf end):
    m_start(start), m_end(end)
{
    //Create an arrow.
    vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New();

    // The x-axis is a vector from start to end
    Vertexf diff = end - start;
    Normalf x_axis(diff);
    double length = diff.length();

    // The Z axis is an arbitrary vecotr cross X
    double arbitrary[3];
    vtkMath::RandomSeed(8775070);
    arbitrary[0] = vtkMath::Random(-10,10);
    arbitrary[1] = vtkMath::Random(-10,10);
    arbitrary[2] = vtkMath::Random(-10,10);
    Vertexf dummy(arbitrary[0], arbitrary[1], arbitrary[2]);

    // Compute other two local base vectors
    Normalf z_axis(x_axis.cross(dummy));
    Normalf y_axis(z_axis.cross(x_axis));

    vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
    matrix->Identity();
    for (unsigned int i = 0; i < 3; i++)
    {
        matrix->SetElement(i, 0, x_axis[i]);
        matrix->SetElement(i, 1, y_axis[i]);
        matrix->SetElement(i, 2, z_axis[i]);
    }

    // Apply the transforms
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Translate(start[0], start[1], start[2]);
    transform->Concatenate(matrix);
    transform->Scale(length, length, length);

    // Transform the polydata
    vtkSmartPointer<vtkTransformPolyDataFilter> transformPD = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformPD->SetTransform(transform);
    transformPD->SetInputConnection(arrowSource->GetOutputPort());

    //Create a mapper and actor for the arrow
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    m_arrowActor = vtkSmartPointer<vtkActor>::New();
    mapper->SetInputConnection(arrowSource->GetOutputPort());
    m_arrowActor->SetUserMatrix(transform->GetMatrix());
    m_arrowActor->SetMapper(mapper);

    // Create spheres for start and end point
    vtkSmartPointer<vtkSphereSource> sphereStartSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereStartSource->SetCenter(start[0], start[1], start[2]);
    vtkSmartPointer<vtkPolyDataMapper> sphereStartMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    sphereStartMapper->SetInputConnection(sphereStartSource->GetOutputPort());
    m_startActor = vtkSmartPointer<vtkActor>::New();
    m_startActor->SetMapper(sphereStartMapper);
    m_startActor->GetProperty()->SetColor(1.0, 1.0, .3);

    vtkSmartPointer<vtkSphereSource> sphereEndSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereEndSource->SetCenter(end[0], end[1], end[2]);
    vtkSmartPointer<vtkPolyDataMapper> sphereEndMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    sphereEndMapper->SetInputConnection(sphereEndSource->GetOutputPort());
    m_endActor = vtkSmartPointer<vtkActor>::New();
    m_endActor->SetMapper(sphereEndMapper);
    m_endActor->GetProperty()->SetColor(1.0, .3, .3);
}

vtkSmartPointer<vtkActor> LVRVtkArrow::getArrowActor()
{
    return m_arrowActor;
}

vtkSmartPointer<vtkActor> LVRVtkArrow::getStartActor()
{
    return m_startActor;
}

vtkSmartPointer<vtkActor> LVRVtkArrow::getEndActor()
{
    return m_endActor;
}

LVRVtkArrow::~LVRVtkArrow()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
