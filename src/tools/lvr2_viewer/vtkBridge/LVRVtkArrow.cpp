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
 * LVRVtkArrow.cpp
 *
 *  @date Feb 21, 2014
 *  @author Thomas Wiemann
 */
#include "LVRVtkArrow.hpp"

#include "lvr2/geometry/Normal.hpp"

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
#include <vtkCubeSource.h>
#include <vtkProperty.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

namespace lvr2
{

LVRVtkArrow::LVRVtkArrow(Vec start, Vec end):
    m_start(start), m_end(end)
{
    //Create an arrow.
    vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New();

    // The x-axis is a vector from start to end
    Vec diff = end - start;
    Normal<float> x_axis(diff);
    double length = diff.length();

    // The Z axis is an arbitrary vecotr cross X
    double arbitrary[3];
    vtkMath::RandomSeed(8775070);
    arbitrary[0] = vtkMath::Random(-10,10);
    arbitrary[1] = vtkMath::Random(-10,10);
    arbitrary[2] = vtkMath::Random(-10,10);
    Vec dummy(arbitrary[0], arbitrary[1], arbitrary[2]);

    // Compute other two local base vectors
    Normal<float> z_axis(x_axis.cross(dummy));
    Normal<float> y_axis(z_axis.cross(x_axis));

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

    //Store default color and set new color to red (the new arrow is active)
    m_arrowActor->GetProperty()->GetColor(m_r, m_g, m_b);
    setTmpColor(1.0, 0.2, 0.2);

    //UNDO THIS
    vtkSmartPointer<vtkCubeSource> cubeStartSource = vtkSmartPointer<vtkCubeSource>::New();
    cubeStartSource->SetBounds(start[0], start[0] + 1, start[1], start[1] + 1,start[2], start[2] + 1);
    vtkSmartPointer<vtkPolyDataMapper> cubeStartMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    cubeStartMapper->SetInputConnection(cubeStartSource->GetOutputPort());

    //UNDO THIS END


    // Create spheres for start and end point
    vtkSmartPointer<vtkSphereSource> sphereStartSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereStartSource->SetCenter(start[0], start[1], start[2]);
    vtkSmartPointer<vtkPolyDataMapper> sphereStartMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    sphereStartMapper->SetInputConnection(sphereStartSource->GetOutputPort());
    m_startActor = vtkSmartPointer<vtkActor>::New();
    //UNDO THIS
    //m_startActor->SetMapper(sphereStartMapper);
    //UNDO THIS END
    m_startActor->SetMapper(cubeStartMapper);
    m_startActor->GetProperty()->SetColor(1.0, 1.0, .3);

    vtkSmartPointer<vtkSphereSource> sphereEndSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereEndSource->SetCenter(end[0], end[1], end[2]);
    vtkSmartPointer<vtkPolyDataMapper> sphereEndMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    sphereEndMapper->SetInputConnection(sphereEndSource->GetOutputPort());
    m_endActor = vtkSmartPointer<vtkActor>::New();
    m_endActor->SetMapper(sphereEndMapper);
    m_endActor->GetProperty()->SetColor(1.0, .3, .3);
}

void LVRVtkArrow::restoreColor()
{
    m_arrowActor->GetProperty()->SetColor(m_r, m_g, m_b);
}

void LVRVtkArrow::setTmpColor(double r, double g, double b)
{
    m_arrowActor->GetProperty()->SetColor(r, g, b);
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

} /* namespace lvr2 */
