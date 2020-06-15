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
 * LVRPickingInteractor.cpp
 *
 *  @date Feb 19, 2014
 *  @author Thomas Wiemann
 */
#include "LVRPickingInteractor.hpp"

#include <vtkObjectFactory.h>
#include <vtkTextProperty.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkAbstractPicker.h>
#include <vtkRenderWindow.h>
#include <vtkRendererCollection.h>
#include <vtkPointPicker.h>
#include <vtkCamera.h>
#include <vtkCallbackCommand.h>
#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkCubeSource.h>
#include <vtkPlanes.h>
#include <vtkAreaPicker.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkVersion.h>
#include <vtkPointData.h>
#include <vtkIdTypeArray.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkPolyLine.h>
#include <vtkObjectFactory.h>
#include <vtkPointSource.h>
#include <vtkExtractGeometry.h>
#include <vtkDataSetMapper.h>
#include <vtkUnstructuredGrid.h>
#include <vtkIdFilter.h>
#include <set>

namespace lvr2
{

vtkStandardNewMacro(LVRPickingInteractor);

LVRPickingInteractor::LVRPickingInteractor() :
//LVRPickingInteractor::LVRPickingInteractor(vtkSmartPointer<vtkRenderer> renderer) :
   m_motionFactor(50), m_rotationFactor(20), m_interactorMode(TRACKBALL)
{
    m_modified = false;
    m_startCameraMovePosition[0] = 0;
    m_startCameraMovePosition[1] = 0;

    m_viewUp[0] = 0.0;
    m_viewUp[1] = 1.0;
    m_viewUp[2] = 0.0;

    m_pickMode = None;
    m_shooterMode = LOOK;
    m_correspondenceMode = false;
    vtkSmartPointer<vtkTextProperty> p = vtkSmartPointer<vtkTextProperty>::New();
    p->SetColor(1.0, 1.0, 0.0);
    p->SetBold(1);
    p->SetShadow(0);

    m_textActor = vtkSmartPointer<vtkTextActor>::New();
    m_textActor->SetDisplayPosition(100, 10);
    m_textActor->SetTextProperty(p);
    m_textActor->SetInput("Pick a point...");
    m_textActor->VisibilityOff();

}


void LVRPickingInteractor::setRenderer(vtkSmartPointer<vtkRenderer> renderer)
{

    m_renderer = renderer;
    m_renderer->AddActor(m_textActor);
    // Create a sphere actor to represent the current focal point
    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetCenter(0.0, 0.0, 0.0);

    double b[6];
    m_renderer->ComputeVisiblePropBounds(b);
    // Set radius to one percent of the largest scene dimension
    double s = std::max(fabs(b[0] - b[1]), std::max(fabs(b[2] - b[3]), fabs(b[4] - b[5]))) * 0.1;
    cout << s << endl;
    sphereSource->SetRadius(s);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(sphereSource->GetOutputPort());

   // m_selectionPoints = vtkSmartPointer<vtkPoints>::New();
    m_sphereActor = vtkSmartPointer<vtkActor>::New();
    m_sphereActor->SetMapper(mapper);
    m_renderer->AddActor(m_sphereActor);

    double focalPoint[3];
    m_renderer->GetActiveCamera()->GetFocalPoint(focalPoint);
    m_sphereActor->SetPosition(focalPoint[0], focalPoint[1], focalPoint[2]);

    this->UseTimersOn();

}
void LVRPickingInteractor::setStereoMode(int state)
{
     vtkRenderWindowInteractor *rwi = this->Interactor;
     if(state == Qt::Checked)
     {
         rwi->GetRenderWindow()->StereoRenderOn();
     }
     else
     {
         rwi->GetRenderWindow()->StereoRenderOff();
     }
     rwi->Render();
}

void LVRPickingInteractor::setFocalPointRendering(int state)
{
    if(state == Qt::Checked)
    {
        m_sphereActor->SetVisibility(true);
    }
    else
    {
        m_sphereActor->SetVisibility(false);
    }
    vtkRenderWindowInteractor *rwi = this->Interactor;
    rwi->Render();
}

void LVRPickingInteractor::updateFocalPoint()
{
    double focalPoint[3];
    m_renderer->GetActiveCamera()->GetFocalPoint(focalPoint);
    m_sphereActor->SetPosition(focalPoint[0], focalPoint[1], focalPoint[2]);
}

void LVRPickingInteractor::setPoints(vtkSmartPointer<vtkPolyData> points) 
{
    if (points)
    {
        /*
        std::cout <<"point_set" << std::endl;
        auto vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
auto a =  points->GetPointData()->GetAbstractArray("OriginalIds");
      if(a == nullptr) 
      {
          std::cout << "array  not found"<< std::endl;
          std::cout << a->GetNumberOfComponents() << std::endl;
      }
      else
      {
          std::cout << "array found"<< std::endl;
          std::cout << a->GetNumberOfComponents() << std::endl;
      }
        vertexFilter->SetInputData(points);
       
        vertexFilter->Update();
        m_points = {};
        std::cout <<"point_set" << std::endl;
        m_points->DeepCopy(vertexFilter->GetOutput());*/
        m_points = points;
        m_selectedPoints = std::vector<bool>(m_points->GetNumberOfPoints(), false);
        m_pointLabels = std::vector<uint16_t>(m_points->GetNumberOfPoints(), 0);
    }
}

void LVRPickingInteractor::setMotionFactor(double factor)
{
    m_motionFactor = factor;
}

void LVRPickingInteractor::setRotationFactor(double factor)
{
    m_rotationFactor = factor;
}

void LVRPickingInteractor::modeTerrain()
{
    m_interactorMode = TERRAIN;
}

void LVRPickingInteractor::modeTrackball()
{
    m_interactorMode = TRACKBALL;
}

void LVRPickingInteractor::modeShooter()
{
    m_interactorMode = SHOOTER;
}

LVRPickingInteractor::~LVRPickingInteractor()
{
    // TODO Auto-generated destructor stub
}


void LVRPickingInteractor::Dolly()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        dollyTrackball();
        break;
    case TERRAIN:
        dollyTerrain();
        break;
    case SHOOTER:
        dollyShooter();
        break;
    default:
        dollyTrackball();
    }
}

void LVRPickingInteractor::Dolly(double speed)
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        dollyTrackball(speed);
        break;
    case TERRAIN:
        dollyTerrain(speed);
        break;
    case SHOOTER:
        dollyShooter(speed);
        break;
    default:
        dollyTrackball();

    }
}

void LVRPickingInteractor::Pan()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        panTrackball();
        break;
    case TERRAIN:
        panTerrain();
        break;
    case SHOOTER:
        panShooter();
        break;
    default:
        panTrackball();

    }
}

void LVRPickingInteractor::Spin()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        spinTrackball();
        break;
    case TERRAIN:
        spinTerrain();
        break;
    case SHOOTER:
        spinShooter();
        break;
    default:
        spinTrackball();
    }
}

void LVRPickingInteractor::Rotate()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        rotateTrackball();
        break;
    case TERRAIN:
        rotateTerrain();
        break;
    case SHOOTER:
        rotateShooter();
        break;
    default:
        rotateTrackball();
    }
}

void LVRPickingInteractor::Zoom()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        zoomTrackball();
        break;
    case TERRAIN:
        zoomTerrain();
        break;
    case SHOOTER:
        zoomShooter();
        break;
    default:
        zoomTrackball();
    }
}


void LVRPickingInteractor::OnLeftButtonDown()
{
    if (m_pickMode == PickLabel)
    {
	LVRInteractorStylePolygonPick::OnLeftButtonDown();
	return;
    } 
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onLeftButtonDownTrackball();
        break;
    case TERRAIN:
        onLeftButtonDownTerrain();
        break;
    case SHOOTER:
        onLeftButtonDownShooter();
        break;
    default:
        onLeftButtonDownTrackball();
    }
    handlePicking();
}


void LVRPickingInteractor::OnLeftButtonUp()
{
    if (m_pickMode == PickLabel)
    {
	calculateSelection(true);
	return;
    } 
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onLeftButtonUpTrackball();
        break;
    case TERRAIN:
        onLeftButtonUpTerrain();
        break;
    case SHOOTER:
        onLeftButtonUpShooter();
        break;
    default:
        onLeftButtonUpTrackball();
    }
}

void LVRPickingInteractor::OnMouseMove()
{
    if(m_pickMode == PickLabel)
    {
        LVRInteractorStylePolygonPick::OnMouseMove();
	return;
    }
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onMouseMoveTrackball();
        break;
    case TERRAIN:
        onMouseMoveTerrain();
        break;
    case SHOOTER:
        //onMouseMoveShooter();
        break;
    default:
        onMouseMoveTrackball();
    }
}

void LVRPickingInteractor::OnMiddleButtonUp()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onMiddleButtonUpTrackball();
        break;
    case TERRAIN:
        onMiddleButtonUpTerrain();
        break;
    case SHOOTER:
        onMiddleButtonUpShooter();
        break;
    default:
        onMiddleButtonUpTrackball();
    }
}

void LVRPickingInteractor::OnMiddleButtonDown()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onMiddleButtonDownTrackball();
        break;
    case TERRAIN:
        onMiddleButtonDownTerrain();
        break;
    case SHOOTER:
        onMiddleButtonDownShooter();
        break;
    default:
        onMiddleButtonDownTrackball();
    }
}

void LVRPickingInteractor::OnRightButtonUp()
{
    if (m_pickMode == PickLabel)
    {
	calculateSelection(false);
	return;
    } 
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onRightButtonUpTrackball();
        break;
    case TERRAIN:
        onRightButtonUpTerrain();
        break;
    case SHOOTER:
        onRightButtonUpShooter();
        break;
    default:
        onRightButtonUpTrackball();
    }
}

void LVRPickingInteractor::OnRightButtonDown()
{
    if (m_pickMode == PickLabel)
    {
	LVRInteractorStylePolygonPick::OnLeftButtonDown();
	return;
    }
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onRightButtonDownTrackball();
        break;
    case TERRAIN:
        onRightButtonDownTerrain();
        break;
    case SHOOTER:
        onRightButtonDownShooter();
        break;
    default:
        onRightButtonDownTrackball();
    }
}

void LVRPickingInteractor::OnMouseWheelBackward()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onMouseWheelBackwardTrackball();
        break;
    case TERRAIN:
        onMouseWheelBackwardTerrain();
        break;
    case SHOOTER:
        onMouseWheelBackwardShooter();
        break;

    default:
        onMouseWheelBackwardTrackball();
    }
}

void LVRPickingInteractor::OnMouseWheelForward()
{
    switch(m_interactorMode)
    {
    case TRACKBALL:
        onMouseWheelForwardTrackball();
        break;
    case TERRAIN:
        onMouseWheelForwardTerrain();
        break;
    case SHOOTER:
        onMouseWheelForwardShooter();
        break;
    default:
        onMouseWheelForwardTrackball();
    }
}

void LVRPickingInteractor::dollyShooter()
{

}

void LVRPickingInteractor::strafeShooter(double factor)
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();


    double position[3];
    double direction[3];
    double viewUp[3];
    double cross[3];
    double focalPoint[3];

    camera->GetPosition(position);
    camera->GetDirectionOfProjection(direction);
    camera->GetFocalPoint(focalPoint);
    camera->GetViewUp(viewUp[0], viewUp[1], viewUp[2]);

    vtkMath::Cross(direction, viewUp, cross);

    // Move position
    camera->SetPosition(
                position[0] + 3 * factor * cross[0],
                position[1] + 3 * factor * cross[1],
                position[2] + 3 * factor * cross[2]);

    // Move position
    camera->SetFocalPoint(
                focalPoint[0] + 3 * factor * cross[0],
                focalPoint[1] + 3 * factor * cross[1],
                focalPoint[2] + 3 * factor * cross[2]);

    camera->OrthogonalizeViewUp();

    if (this->AutoAdjustCameraClippingRange)
    {
        this->CurrentRenderer->ResetCameraClippingRange();
    }


    rwi->Render();
}

void LVRPickingInteractor::resetViewUpShooter()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    camera->SetViewUp(m_viewUp[0], m_viewUp[1], m_viewUp[2]);
    rwi->Render();
}

void LVRPickingInteractor::resetCamera()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    if(this->CurrentRenderer!=nullptr)
    {
        this->CurrentRenderer->ResetCamera();
    }
    else
    {
        vtkWarningMacro(<<"no current renderer on the interactor style.");
    }
    rwi->Render();
}

void LVRPickingInteractor::dollyShooter(double factor)
{

    vtkRenderWindowInteractor *rwi = this->Interactor;
    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();

    double position[3];
    double direction[3];

    camera->GetPosition(position);
    camera->GetDirectionOfProjection(direction);

    camera->SetFocalPoint(
                position[0] + 3 * fabs(factor) * direction[0],
                position[1] + 3 * fabs(factor) * direction[1],
                position[2] + 3 * fabs(factor) * direction[2]);



    // Compute desired position
    camera->SetPosition(
                position[0] - factor * direction[0],
                position[1] - factor * direction[1],
                position[2] - factor * direction[2]);

    camera->OrthogonalizeViewUp();

    if (this->AutoAdjustCameraClippingRange)
    {
        this->CurrentRenderer->ResetCameraClippingRange();
    }

    rwi->Render();
}

void LVRPickingInteractor::onMouseMoveShooter()
{

    vtkRenderWindowInteractor *rwi = this->Interactor;
    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();

    // Compute offset from screen center
    int mx = rwi->GetSize()[0] / 2;
    int my = rwi->GetSize()[1] / 2;
    int dx = rwi->GetEventPosition()[0] - m_startCameraMovePosition[0];
    int dy = rwi->GetEventPosition()[1] - m_startCameraMovePosition[1];


    float yawAngle = (1.0 * dx / mx) * m_rotationFactor;
    float pitchAngle =  (1.0 * dy / my) * m_rotationFactor;

    camera->Yaw(-yawAngle);
    camera->Pitch(pitchAngle);
    camera->OrthogonalizeViewUp();
    camera->ViewingRaysModified();

    if (this->AutoAdjustCameraClippingRange)
    {
        this->CurrentRenderer->ResetCameraClippingRange();
    }

    rwi->Render();
}

void LVRPickingInteractor::hoverShooter()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();

    int my = rwi->GetSize()[1] / 2;
    int dy = rwi->GetEventPosition()[1] - m_startCameraMovePosition[1];

    double step = -(1.0 * dy / my) * m_motionFactor;
    double position[3];
    double focal[3];
    double direction[3];

    camera->GetPosition(position);
    camera->GetFocalPoint(focal);
    camera->GetViewUp(direction);

//    cout << "STEP: " << step << endl;
//    cout << "POSITION: " << position[0] << " " << position[1] << " " << position[2] << endl;
//    cout << "DIRECTION: " << direction[0] << " " << direction[1] << " " << direction[2] << endl;

    camera->SetPosition(
                position[0] + step * direction[0],
                position[1] + step * direction[1],
                position[2] + step * direction[2]);

    // Move position
    camera->SetFocalPoint(
                focal[0] + step * direction[0],
                focal[1] + step * direction[1],
                focal[2] + step * direction[2]);


    camera->ViewingRaysModified();

    if (this->AutoAdjustCameraClippingRange)
    {
        this->CurrentRenderer->ResetCameraClippingRange();
    }

    rwi->Render();

}


void LVRPickingInteractor::panShooter()
{

}

void LVRPickingInteractor::spinShooter()
{

}

void LVRPickingInteractor::zoomShooter()
{

}

void LVRPickingInteractor::rotateShooter()
{

}

void LVRPickingInteractor::onLeftButtonDownShooter()
{
    this->m_shooterMode = LOOK;
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    GrabFocus(this->EventCallbackCommand, nullptr);
    this->StartTimer();

    // Save klicked position to determine rotation speed
    vtkRenderWindowInteractor *rwi = this->Interactor;
    m_startCameraMovePosition[0] = rwi->GetEventPosition()[0];
    m_startCameraMovePosition[1] = rwi->GetEventPosition()[1];
}

void LVRPickingInteractor::onLeftButtonUpShooter()
{
    switch (this->State)
    {
    case VTKIS_TIMER:
        this->EndTimer();
        if ( this->Interactor )
        {
            this->ReleaseFocus();
        }
        break;
    }
}

void LVRPickingInteractor::OnTimer()
{
    if(m_interactorMode == SHOOTER)
    {
        if(m_shooterMode == LOOK)
        {
            onMouseMoveShooter();
        }
        if(m_shooterMode == HOVER)
        {
            hoverShooter();
        }
    }
}



void LVRPickingInteractor::onMiddleButtonUpShooter()
{
    switch (this->State)
    {
    case VTKIS_TIMER:
        this->EndTimer();
        if ( this->Interactor )
        {
            this->ReleaseFocus();
        }
        break;
    }
}

void LVRPickingInteractor::onMiddleButtonDownShooter()
{
    this->m_shooterMode = HOVER;
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    GrabFocus(this->EventCallbackCommand, nullptr);
    this->StartTimer();

    // Save klicked position to determine rotation speed
    vtkRenderWindowInteractor *rwi = this->Interactor;
    m_startCameraMovePosition[0] = rwi->GetEventPosition()[0];
    m_startCameraMovePosition[1] = rwi->GetEventPosition()[1];
}

void LVRPickingInteractor::onRightButtonUpShooter()
{

}

void LVRPickingInteractor::onRightButtonDownShooter()
{

}

void LVRPickingInteractor::onMouseWheelBackwardShooter()
{

}

void LVRPickingInteractor::onMouseWheelForwardShooter()
{

}

void LVRPickingInteractor::dollyTerrain()
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkRenderWindowInteractor *rwi = this->Interactor;
    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    double *center = this->CurrentRenderer->GetCenter();

    int dy = rwi->GetEventPosition()[1] - rwi->GetLastEventPosition()[1];
    double dyf = m_motionFactor * dy / center[1];
    double zoomFactor = pow(1.1, dyf);

    if (camera->GetParallelProjection())
    {
        camera->SetParallelScale(camera->GetParallelScale() / zoomFactor);
    }
    else
    {
        camera->Dolly( zoomFactor );
        if (this->AutoAdjustCameraClippingRange)
        {
            this->CurrentRenderer->ResetCameraClippingRange();
        }
    }

    if (rwi->GetLightFollowCamera())
    {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    rwi->Render();
}

void LVRPickingInteractor::dollyTerrain(double factor)
{

}

void LVRPickingInteractor::panTerrain()
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkRenderWindowInteractor *rwi = this->Interactor;

    // Get the vector of motion

    double fp[3], focalPoint[3], pos[3], v[3], p1[4], p2[4];

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    camera->GetPosition( pos );
    camera->GetFocalPoint( fp );

    this->ComputeWorldToDisplay(fp[0], fp[1], fp[2],
            focalPoint);

    this->ComputeDisplayToWorld(rwi->GetEventPosition()[0],
            rwi->GetEventPosition()[1],
            focalPoint[2],
            p1);

    this->ComputeDisplayToWorld(rwi->GetLastEventPosition()[0],
            rwi->GetLastEventPosition()[1],
            focalPoint[2],
            p2);

    for (int i=0; i<3; i++)
    {
        v[i] = p2[i] - p1[i];
        pos[i] += v[i];
        fp[i] += v[i];
    }

    camera->SetPosition( pos );
    camera->SetFocalPoint( fp );

    if (rwi->GetLightFollowCamera())
    {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    rwi->Render();
}

void LVRPickingInteractor::spinTerrain()
{

}

void LVRPickingInteractor::zoomTerrain()
{

}

void LVRPickingInteractor::rotateTerrain()
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkRenderWindowInteractor *rwi = this->Interactor;

    int dx = - ( rwi->GetEventPosition()[0] - rwi->GetLastEventPosition()[0] );
    int dy = - ( rwi->GetEventPosition()[1] - rwi->GetLastEventPosition()[1] );

    int *size = this->CurrentRenderer->GetRenderWindow()->GetSize();

    double a = dx / static_cast<double>( size[0]) * 180.0;
    double e = dy / static_cast<double>( size[1]) * 180.0;

    if (rwi->GetShiftKey())
    {
        if(abs( dx ) >= abs( dy ))
        {
            e = 0.0;
        }
        else
        {
            a = 0.0;
        }
    }

    // Move the camera.
    // Make sure that we don't hit the north pole singularity.

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    camera->Azimuth( a );

    double dop[3], vup[3];

    camera->GetDirectionOfProjection( dop );
    vtkMath::Normalize( dop );
    camera->GetViewUp( vup );
    vtkMath::Normalize( vup );

    double angle = vtkMath::DegreesFromRadians( acos(vtkMath::Dot( dop, vup) ) );
    if ( ( angle + e ) > 179.0 ||
         ( angle + e ) < 1.0 )
    {
        e = 0.0;
    }

    camera->Elevation( e );

    if ( this->AutoAdjustCameraClippingRange )
    {
        this->CurrentRenderer->ResetCameraClippingRange();
    }

    rwi->Render();
}

void LVRPickingInteractor::onLeftButtonDownTerrain()
{
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartRotate();
}

void LVRPickingInteractor::onLeftButtonUpTerrain()
{
    switch (this->State)
    {
    case VTKIS_ROTATE:
        this->EndRotate();
        if ( this->Interactor )
        {
            this->ReleaseFocus();
        }
        break;
    }
}

void LVRPickingInteractor::onMouseMoveTerrain()
{
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];

    switch (this->State)
    {
    case VTKIS_ROTATE:
        this->FindPokedRenderer(x, y);
        this->Rotate();
        this->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
        break;

    case VTKIS_PAN:
        this->FindPokedRenderer(x, y);
        this->Pan();
        this->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
        break;

    case VTKIS_DOLLY:
        this->FindPokedRenderer(x, y);
        this->Dolly();
        this->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
        break;
    }
}

void LVRPickingInteractor::onMiddleButtonUpTerrain()
{
    switch (this->State)
    {
    case VTKIS_PAN:
        this->EndPan();
        if ( this->Interactor )
        {
            this->ReleaseFocus();
        }
        break;
    }
}

void LVRPickingInteractor::onMiddleButtonDownTerrain()
{
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartPan();
}

void LVRPickingInteractor::onRightButtonUpTerrain()
{
    switch (this->State)
    {
    case VTKIS_DOLLY:
        this->EndDolly();
        if ( this->Interactor )
        {
            this->ReleaseFocus();
        }
        break;
    }
}

void LVRPickingInteractor::onRightButtonDownTerrain()
{
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartDolly();
}

void LVRPickingInteractor::onMouseWheelBackwardTerrain()
{

}

void LVRPickingInteractor::onMouseWheelForwardTerrain()
{

}

void LVRPickingInteractor::dollyTrackball(double factor)
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    if (camera->GetParallelProjection())
    {
        camera->SetParallelScale(camera->GetParallelScale() / factor);
    }
    else
    {
        double position[3];
        double direction[3];

        camera->GetPosition(position);
        camera->GetDirectionOfProjection(direction);

        // Compute desired position
        camera->SetPosition(
                    position[0] - factor * direction[0],
                    position[1] - factor * direction[1],
                    position[2] - factor * direction[2]);

        if (this->AutoAdjustCameraClippingRange)
        {
            this->CurrentRenderer->ResetCameraClippingRange();
        }
    }

    if (this->Interactor->GetLightFollowCamera())
    {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    this->Interactor->Render();
}

void LVRPickingInteractor::dollyTrackball()
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkRenderWindowInteractor *rwi = this->Interactor;
    double *center = this->CurrentRenderer->GetCenter();
    int dy = rwi->GetEventPosition()[1] - rwi->GetLastEventPosition()[1];
    double dyf = 1 * dy / center[1];

    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    if (camera->GetParallelProjection())
    {
        camera->SetParallelScale(camera->GetParallelScale() / dyf);
    }
    else
    {
        camera->Dolly(dyf);
        if (this->AutoAdjustCameraClippingRange)
        {
            this->CurrentRenderer->ResetCameraClippingRange();
        }
    }

    if (this->Interactor->GetLightFollowCamera())
    {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    this->Interactor->Render();
}

void LVRPickingInteractor::panTrackball()
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkRenderWindowInteractor *rwi = this->Interactor;

    double viewFocus[4], focalDepth, viewPoint[3];
    double newPickPoint[4], oldPickPoint[4], motionVector[3];

    // Calculate the focal depth since we'll be using it a lot

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    camera->GetFocalPoint(viewFocus);
    this->ComputeWorldToDisplay(viewFocus[0], viewFocus[1], viewFocus[2],
            viewFocus);
    focalDepth = viewFocus[2];

    this->ComputeDisplayToWorld(rwi->GetEventPosition()[0],
            rwi->GetEventPosition()[1],
            focalDepth,
            newPickPoint);

    // Has to recalc old mouse point since the viewport has moved,
    // so can't move it outside the loop

    this->ComputeDisplayToWorld(rwi->GetLastEventPosition()[0],
            rwi->GetLastEventPosition()[1],
            focalDepth,
            oldPickPoint);

    // Camera motion is reversed

    motionVector[0] = oldPickPoint[0] - newPickPoint[0];
    motionVector[1] = oldPickPoint[1] - newPickPoint[1];
    motionVector[2] = oldPickPoint[2] - newPickPoint[2];

    camera->GetFocalPoint(viewFocus);
    camera->GetPosition(viewPoint);
    camera->SetFocalPoint(motionVector[0] + viewFocus[0],
            motionVector[1] + viewFocus[1],
            motionVector[2] + viewFocus[2]);

    camera->SetPosition(motionVector[0] + viewPoint[0],
            motionVector[1] + viewPoint[1],
            motionVector[2] + viewPoint[2]);

    if (rwi->GetLightFollowCamera())
    {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    rwi->Render();
}

void LVRPickingInteractor::spinTrackball()
{
    if ( this->CurrentRenderer == nullptr )
    {
        return;
    }

    vtkRenderWindowInteractor *rwi = this->Interactor;

    double *center = this->CurrentRenderer->GetCenter();

    double newAngle =
            vtkMath::DegreesFromRadians( atan2( rwi->GetEventPosition()[1] - center[1],
            rwi->GetEventPosition()[0] - center[0] ) );

    double oldAngle =
            vtkMath::DegreesFromRadians( atan2( rwi->GetLastEventPosition()[1] - center[1],
            rwi->GetLastEventPosition()[0] - center[0] ) );

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    camera->Roll( newAngle - oldAngle );
    camera->OrthogonalizeViewUp();

    rwi->Render();
}

void LVRPickingInteractor::zoomTrackball()
{
    this->Zoom();
}

void LVRPickingInteractor::rotateTrackball()
{
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    vtkRenderWindowInteractor *rwi = this->Interactor;

    int dx = rwi->GetEventPosition()[0] - rwi->GetLastEventPosition()[0];
    int dy = rwi->GetEventPosition()[1] - rwi->GetLastEventPosition()[1];

    int *size = this->CurrentRenderer->GetRenderWindow()->GetSize();

    double delta_elevation = -m_rotationFactor / size[1];
    double delta_azimuth = -m_rotationFactor / size[0];

    double rxf = dx * delta_azimuth * m_motionFactor;
    double ryf = dy * delta_elevation * m_motionFactor;

    vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
    camera->Azimuth(rxf);
    camera->Elevation(ryf);
    camera->OrthogonalizeViewUp();

    if (this->AutoAdjustCameraClippingRange)
    {
        this->CurrentRenderer->ResetCameraClippingRange();
    }

    if (rwi->GetLightFollowCamera())
    {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    rwi->Render();
}

void LVRPickingInteractor::correspondenceSearchOn()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    m_correspondenceMode = true;
    m_textActor->SetInput("Pick first correspondence point...");
    m_textActor->VisibilityOn();
    rwi->Render();
    m_pickMode = PickFirst;
}

void LVRPickingInteractor::correspondenceSearchOff()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    m_correspondenceMode = false;
    m_pickMode = None;
    m_textActor->VisibilityOff();
    rwi->Render();
}

void LVRPickingInteractor::labelingOn()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    m_textActor->SetInput("Press \"l\" to select Points");
    m_textActor->VisibilityOn();
    rwi->Render();

    m_labelingMode = true;
    labelModeChanged(m_pickMode == PickLabel);
    //m_pickMode = PickLabel;

}
void LVRPickingInteractor::labelingOff()
{
        vtkRenderWindowInteractor *rwi = this->Interactor;
	m_labelingMode = false;
        m_pickMode = None;
	m_textActor->VisibilityOff();
        rwi->Render();
}
void LVRPickingInteractor::handlePicking()
{
    vtkPointPicker* picker = (vtkPointPicker*)this->Interactor->GetPicker();

    if(m_pickMode == None)
    {
        this->m_numberOfClicks++;
        //std::cout << "m_numberOfClicks = " << this->m_numberOfClicks << std::endl;
        int pickPosition[2];
        this->GetInteractor()->GetEventPosition(pickPosition);

        int xdist = pickPosition[0] - this->m_previousPosition[0];
        int ydist = pickPosition[1] - this->m_previousPosition[1];

        this->m_previousPosition[0] = pickPosition[0];
        this->m_previousPosition[1] = pickPosition[1];

        int moveDistance = (int)sqrt((double)(xdist*xdist + ydist*ydist));

        // Reset numClicks - If mouse moved further than resetPixelDistance
        if(moveDistance > 5)
        {
            this->m_numberOfClicks = 1;
        }

        if(this->m_numberOfClicks == 2)
        {
            this->m_numberOfClicks = 0;
            double* picked = new double[3];
            picker->Pick(pickPosition[0],
                    pickPosition[1],
                    0,  // always zero.
                    this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer());
            vtkActor* actor = picker->GetActor();
            int point = picker->GetPointId();
            Q_EMIT(pointSelected(actor, point));
        }
    }
    else if(m_pickMode == PickFocal)
    {
        int* pickPos = this->Interactor->GetEventPosition();
        double* picked = new double[3];
        int res = picker->Pick(this->Interactor->GetEventPosition()[0],
                this->Interactor->GetEventPosition()[1],
                0,  // always zero.
                this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer());

        if(res)
        {
            picker->GetPickPosition(picked);

            vtkCamera* cam = m_renderer->GetActiveCamera();
            cam->SetFocalPoint(picked[0], picked[1], picked[2]);
            updateFocalPoint();

            m_textActor->VisibilityOff();
        }
        m_pickMode = None;
    }
    else if(m_pickMode == PickLabel)
    {


    }
    else
    {
        int* pickPos = this->Interactor->GetEventPosition();
        double* picked = new double[3];
        picker->Pick(this->Interactor->GetEventPosition()[0],
                this->Interactor->GetEventPosition()[1],
                0,  // always zero.
                this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer());
        picker->GetPickPosition(picked);

        if(m_pickMode == PickFirst)
        {
            Q_EMIT(firstPointPicked(picked));
        }
        else if(m_pickMode == PickSecond)
        {
            Q_EMIT(secondPointPicked(picked));
        }
    }
}
void LVRPickingInteractor::onLeftButtonDownTrackball()
{
    // Code taken from vtkInteractorStyleTrackballCamera
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }
    if(m_labelingMode)
    {
	  //  vtkInteractorStyleRubberBandPick::OnLeftButtonDown();
	    //return;
    }

    GrabFocus(this->EventCallbackCommand, nullptr);
    if (this->Interactor->GetShiftKey())
    {
        if (this->Interactor->GetControlKey())
        {
            this->StartDolly();
        }
        else
        {
            this->StartPan();
        }
    }
    else
    {
        if (this->Interactor->GetControlKey())
        {
            this->StartSpin();
        }
        else
        {
            this->StartRotate();
        }
    }
}

void LVRPickingInteractor::onLeftButtonUpTrackball()
{
    switch (this->State)
    {
    case VTKIS_DOLLY:
        this->EndDolly();
        break;

    case VTKIS_PAN:
        this->EndPan();
        break;

    case VTKIS_SPIN:
        this->EndSpin();
        break;

    case VTKIS_ROTATE:
        this->EndRotate();
        break;
    }

    if ( this->Interactor )
    {
        this->ReleaseFocus();
    }
}

void LVRPickingInteractor::onMouseMoveTrackball()
{
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];

    switch (this->State)
    {
    case VTKIS_ROTATE:
        this->FindPokedRenderer(x, y);
        this->Rotate();
        this->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
        break;

    case VTKIS_PAN:
        this->FindPokedRenderer(x, y);
        this->Pan();
        this->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
        break;

    case VTKIS_DOLLY:
        this->FindPokedRenderer(x, y);
        this->Dolly();
        this->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
        break;

    case VTKIS_SPIN:
        this->FindPokedRenderer(x, y);
        this->Spin();
        this->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
        break;
    }
}

void LVRPickingInteractor::onMiddleButtonUpTrackball()
{
    switch (this->State)
    {
    case VTKIS_PAN:
        this->EndPan();
        if ( this->Interactor )
        {
            this->ReleaseFocus();
        }
        break;
    }
}

void LVRPickingInteractor::onMouseWheelForwardTrackball()
{
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartDolly();
    double factor = m_motionFactor * this->MouseWheelMotionFactor;
    this->Dolly(-pow(1.1, factor));
    this->EndDolly();
    this->ReleaseFocus();
}

void LVRPickingInteractor::onMouseWheelBackwardTrackball()
{
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartDolly();
    double factor = m_motionFactor * this->MouseWheelMotionFactor;
    this->Dolly(pow(1.1, factor));
    this->EndDolly();
    this->ReleaseFocus();
}

void LVRPickingInteractor::onMiddleButtonDownTrackball()
{
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartPan();
}

void LVRPickingInteractor::onRightButtonUpTrackball()
{
    switch (this->State)
    {
    case VTKIS_DOLLY:
        this->EndDolly();

        if ( this->Interactor )
        {
            this->ReleaseFocus();
        }
        break;
    }
}

void LVRPickingInteractor::onRightButtonDownTrackball()
{
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == nullptr)
    {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartDolly();
}


void LVRPickingInteractor::OnKeyPress()
{

}

void LVRPickingInteractor::OnKeyDown()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    std::string key = rwi->GetKeySym();

    if(key == "Escape" && m_labelingMode)
    {
	discardChanges();
    }
    if(key == "Return" && m_labelingMode)
    {
	LVRInteractorStylePolygonPick::OnKeyDown();
        if (m_pickMode == PickLabel && !lassoToolSelected)
        {
    	    calculateSelection(true);
        } 
	saveCurrentLabelSelection();
    }
    if(key == "x" && m_correspondenceMode)
    {
        m_textActor->SetInput("Pick first correspondence point...");
        m_textActor->VisibilityOn();
        rwi->Render();
        m_pickMode = PickFirst;
    }

    if(key == "y" && m_correspondenceMode)
    {
        m_textActor->SetInput("Pick second correspondence point...");
        m_textActor->VisibilityOn();
        rwi->Render();
        m_pickMode = PickSecond;
    }
    if(m_labelingMode && key == "l")
    {
        labelModeChanged(m_pickMode == PickLabel);
    }

    if(key == "f")
    {
        pickFocalPoint();
    }

    if(key == "q")
    {
        m_textActor->VisibilityOff();
        m_pickMode = None;
        rwi->Render();
    }


}

void LVRPickingInteractor::OnChar()
{
    if(m_pickMode == PickLabel)
    {
	
        LVRInteractorStylePolygonPick::OnChar();
	return;
    }
    if(m_interactorMode == SHOOTER)
    {
        vtkRenderWindowInteractor *rwi = this->Interactor;
        vtkCamera* cam = this->CurrentRenderer->GetActiveCamera();
        switch (rwi->GetKeyCode())
        {
        case 'w':
        case 'W':
            dollyShooter(-this->m_motionFactor);
            break;
        case 'a':
        case 'A':
            strafeShooter(-this->m_motionFactor);
            break;
        case 's':
        case 'S':
            dollyShooter(this->m_motionFactor);
            break;
        case 'd':
        case 'D':
            strafeShooter(this->m_motionFactor);
            cout << "D" << endl;
            break;
        case 'u':
        case 'U':
            resetViewUpShooter();
            break;
        case '1':
            m_viewUp[0] = 1.0;
            m_viewUp[1] = 0.0;
            m_viewUp[2] = 0.0;
            resetViewUpShooter();
            break;
        case '2':
            m_viewUp[0] = 0.0;
            m_viewUp[1] = 1.0;
            m_viewUp[2] = 0.0;
            resetViewUpShooter();
            break;
        case '3':
            m_viewUp[0] = 0.0;
            m_viewUp[1] = 0.0;
            m_viewUp[2] = 1.0;
            resetViewUpShooter();
            break;
        case 'R':
        case 'r':
            resetCamera();
            break;
        }  
    }

    else
    {
        vtkInteractorStyle::OnChar();
    }
}

void LVRPickingInteractor::pickFocalPoint()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    m_textActor->SetInput("Pick new camera focal point...");
    m_textActor->VisibilityOn();
    rwi->Render();
    m_pickMode = PickFocal;
}

void LVRPickingInteractor::OnKeyRelease()
{

}

bool LVRPickingInteractor::isInside(std::vector<vtkVector2i>* polygon, int& pX, int& pY)
{
    int n = polygon->size();
    if (n < 3) return false;

    int i, j, c = 0;
    for(i = 0, j = n -1; i < n ; j = i++)
    {
	auto& vert = polygon->at(j);
	auto& vert_next = polygon->at(i);

	if (((vert_next.GetY() > pY) != (vert.GetY() > pY)) &&
		(pX < (vert.GetX() - vert_next.GetX()) * (pY - vert_next.GetY()) / (vert.GetY() - vert_next.GetY()) + vert_next.GetX()))
	{
	    c = !c;
        }
    }
    return c;
}


void LVRPickingInteractor::calculateSelection(bool select)
{

      // Forward events
      LVRInteractorStylePolygonPick::OnLeftButtonUp();

      if (!m_points)
      {
	      return;
      }

      if (m_labelActors.find(m_selectedLabel) != m_labelActors.end())
      {
      	this->CurrentRenderer->RemoveActor(m_labelActors[m_selectedLabel]);
      }

      this->CurrentRenderer->RemoveActor(m_selectedActor);
      m_selectedActor = vtkSmartPointer<vtkActor>::New();
      m_selectedMapper = vtkSmartPointer<vtkDataSetMapper>::New();
      m_selectedActor->SetMapper(m_selectedMapper);
  
      vtkPlanes* frustum = static_cast<vtkAreaPicker*>(this->GetInteractor()->GetPicker())->GetFrustum();

      vtkSmartPointer<vtkExtractGeometry> extractGeometry =
        vtkSmartPointer<vtkExtractGeometry>::New();
      extractGeometry->SetImplicitFunction(frustum);
#if VTK_MAJOR_VERSION <= 5
      extractGeometry->SetInput(m_points);
#else
      extractGeometry->SetInputData(m_points);
#endif
      extractGeometry->Update();

      vtkSmartPointer<vtkVertexGlyphFilter> glyphFilter =
        vtkSmartPointer<vtkVertexGlyphFilter>::New();
      glyphFilter->SetInputConnection(extractGeometry->GetOutputPort());
      glyphFilter->Update();

      vtkPolyData* selected = glyphFilter->GetOutput();
      
      vtkIdTypeArray* ids = vtkIdTypeArray::SafeDownCast(selected->GetPointData()->GetArray("OriginalIds"));
      m_selectedIds = vtkIdTypeArray::SafeDownCast(selected->GetPointData()->GetArray("OriginalIds"));
 
      //TODO Check if smart Pointer realy necassary
      vtkSmartPointer<vtkCoordinate> coordinate = vtkSmartPointer<vtkCoordinate>::New();
      coordinate->SetCoordinateSystemToWorld();

      std::vector<vtkVector2i> polygonPoints = this->GetPolygonPoints();
      std::vector<int> selectedPolyPoints;
      selectedPolyPoints.resize(ids->GetNumberOfTuples());

      for (vtkIdType i = 0; i < ids->GetNumberOfTuples(); i++)
      {
	auto selectedPoint = m_points->GetPoint(ids->GetValue(i));
	coordinate->SetValue(selectedPoint[0], selectedPoint[1], selectedPoint[2]); 
	int* displayCoord;
	displayCoord = coordinate->GetComputedViewportValue(this->CurrentRenderer);
	if (isInside(&polygonPoints, displayCoord[0], displayCoord[1]))
	{
		selectedPolyPoints.push_back(ids->GetValue(i));
	}
      }

#if VTK_MAJOR_VERSION <= 5
      m_selectedMapper->SetInput(selected);
#else
      m_selectedMapper->SetInputData(selected);
#endif
      m_selectedMapper->ScalarVisibilityOff();

      for(auto selectedPolyPoint : selectedPolyPoints)
      {
		m_selectedPoints[selectedPolyPoint] = select;
      }

      auto vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
      auto selectedVtkPoints = vtkSmartPointer<vtkPoints>::New();

      double point[3];
      for (int i = 0; i < m_selectedPoints.size(); i++)
      {
	      if (m_selectedPoints[i])
	      {
      		m_points->vtkDataSet::GetPoint(i, point);
      		selectedVtkPoints->InsertNextPoint(point);
	      }
      }
      if (selectedPolyPoints.size() > 0)
      {
	    m_modified = true;
      }

      auto selectedVtkPoly = vtkSmartPointer<vtkPolyData>::New();
      selectedVtkPoly->SetPoints(selectedVtkPoints);


      vertexFilter->SetInputData(selectedVtkPoly);
      vertexFilter->Update();

      auto polyData = vtkSmartPointer<vtkPolyData>::New();
      polyData->ShallowCopy(vertexFilter->GetOutput());

#if VTK_MAJOR_VERSION <= 5
      m_selectedMapper->SetInput(polyData);
#else
      m_selectedMapper->SetInputData(polyData);
#endif
      m_selectedMapper->ScalarVisibilityOff();   


      int r, g, b;
      if(!m_labelColors.empty())
      {
        m_labelColors[m_selectedLabel].getRgb(&r, &g, &b);
        m_selectedActor->GetProperty()->SetColor(r/255.0, g/255.0, b/255.0); //(R,G,B)
      }
      //SelectedActor->GetProperty()->SetPointSize(3);

      m_renderer->AddActor(m_selectedActor);
      this->GetInteractor()->GetRenderWindow()->Render();
      this->HighlightProp(NULL);

}

void LVRPickingInteractor::newLabel(QTreeWidgetItem* item)
{
    m_labelColors[item->data(3,0).toInt()] =  item->data(3,1).value<QColor>();
    if(m_labelColors.size() == 1)
    {
            //first Label set as the choosen label
            m_selectedLabel = item->data(3,0).toInt();
    }
    if(m_labelActors.find(item->data(3,0).toInt()) != m_labelActors.end())
    {
        //Instance known just change color
        int r,g,b;
        m_labelColors[item->data(3,0).toInt()].getRgb(&r, &g, &b);
        m_labelActors[item->data(3,0).toInt()]->GetProperty()->SetColor(r/255.0, g/255.0, b/255.0); //(R,G,B)
        this->GetInteractor()->GetRenderWindow()->Render();
        this->HighlightProp(NULL);
    }
    /*
    if (item->data(3,0).toInt() == 0)
    {
        //Add Actor for all Unlabeled Points
        auto unlabeledMapper = vtkSmartPointer<vtkDataSetMapper>::New();
        auto unlabeledActor = vtkSmartPointer<vtkActor>::New();
        unlabeledActor->SetMapper(unlabeledMapper);

        //Crete copy of points
        double point[3];
        auto selectedVtkPoints = vtkSmartPointer<vtkPoints>::New();
        for (int i = 0; i < m_selectedPoints.size(); i++)
        {
            m_points->vtkDataSet::GetPoint(i, point);
            selectedVtkPoints->InsertNextPoint(point);
        }
 
        auto selectedVtkPoly = vtkSmartPointer<vtkPolyData>::New();
        selectedVtkPoly->SetPoints(selectedVtkPoints);
        auto vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        vertexFilter->SetInputData(selectedVtkPoly);
        vertexFilter->Update();

        auto polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData->ShallowCopy(vertexFilter->GetOutput());


#if VTK_MAJOR_VERSION <= 5
        unlabeledMapper->SetInput(polyData);
#else
        unlabeledMapper->SetInputData(polyData);
#endif
        unlabeledMapper->ScalarVisibilityOff();   

        unlabeledActor->GetProperty()->SetColor(1,0.0,0.0); //(R,G,B)

        if(m_labelActors.find(0) != m_labelActors.end())
        {
            this->CurrentRenderer->RemoveActor(m_labelActors[0]);
            std::cout << "remove actor bedore" << std::endl;
        }
        m_labelActors[0] = unlabeledActor;

        m_renderer->AddActor(unlabeledActor);
    }*/
}

void LVRPickingInteractor::saveCurrentLabelSelection()
{
    int count = 0;
    std::set<uint16_t> modifiedActors;
    for (int i = 0; i < m_selectedPoints.size(); i++)
    {
        //Set Current selection as new selection for the selected label 
        if(m_selectedPoints[i])
        {
            if(m_pointLabels[i] != m_selectedLabel)
            {
                //Only update if necessary
                modifiedActors.insert(m_pointLabels[i]); 
                m_pointLabels[i] = m_selectedLabel;
            }
            count++;
        }
        else
        {
            //Check if Points were labeled before by this label
            if (m_pointLabels[i] == m_selectedLabel)
            {
                //Set to unlabeled
                m_pointLabels[i] = 0;
                modifiedActors.insert(0); 
            }
        }
    }

    if(m_labelActors.find(m_selectedLabel) != m_labelActors.end())
    {
        this->CurrentRenderer->RemoveActor(m_labelActors[m_selectedLabel]);
    }
    if (modifiedActors.size() > 0)
    {
        m_modified = false;
    }
    Q_EMIT(pointsLabeled(m_selectedLabel, count));

    for (auto modifiedActorLabel : modifiedActors)
    {
        //redraw Actor
        updateActor(modifiedActorLabel);
    }

    //Save Current Actor
    m_labelActors[m_selectedLabel] = m_selectedActor;

    //create new Selection Actor
    m_selectedActor = vtkSmartPointer<vtkActor>::New();
}

void LVRPickingInteractor::discardChanges()
{
	    //Undo Changes 
	    m_renderer->RemoveActor(m_selectedActor);
	    if(m_labelActors.find(m_selectedLabel) != m_labelActors.end())
	    {
		for (int i = 0; i < m_pointLabels.size(); i++)
		{
			m_selectedPoints[i] = (m_pointLabels[i] == m_selectedLabel);

		}
	        m_renderer->AddActor(m_labelActors[m_selectedLabel]);
      		this->GetInteractor()->GetRenderWindow()->Render();
   		this->HighlightProp(NULL);

	    } else
	    {
	       //Label has no selected points just reset selected and delete actor
		std::fill(m_selectedPoints.begin(), m_selectedPoints.end(), 0);
	    	m_selectedActor = vtkSmartPointer<vtkActor>::New();
	    }
	    m_modified = false;
}
void LVRPickingInteractor::labelSelected(uint16_t newLabel)
{
	if(m_selectedLabel == newLabel)
	{
		return;
	}
	if (m_modified)
	{
		QMessageBox acceptDialog;
		acceptDialog.setText("The Label has been modified. Do you want to save your changes?");
		acceptDialog.setStandardButtons(QMessageBox::Save | QMessageBox::Discard);
		acceptDialog.setDefaultButton(QMessageBox::Save);
		int returnValue = acceptDialog.exec();
		switch(returnValue)
		{
			case QMessageBox::Save:
				saveCurrentLabelSelection();
				break;
			case QMessageBox::Discard:
				discardChanges();
				break;
		}
		
	}
	m_modified = false;
	m_selectedLabel = newLabel;
	//TODO ask if Changes should be safed
	
	
	for (int i = 0; i < m_pointLabels.size(); i++)
	{
		m_selectedPoints[i] = (m_pointLabels[i] == newLabel);

	}
}

void LVRPickingInteractor::requestLabels()
{
    Q_EMIT(responseLabels(m_pointLabels));
}
void LVRPickingInteractor::setLabeledPointVisibility(int id, bool visibility)
{
    if(m_labelActors.find(id) == m_labelActors.end())
    {
        //TODO Add new empty actor and set visibility so the choice is kept
        return;
    }

    m_labelActors[id]->SetVisibility(visibility);
    vtkRenderWindowInteractor *rwi = this->Interactor;
    rwi->Render();
}

void LVRPickingInteractor::setLassoTool(bool lassoToolSelected)
{
    if (lassoToolSelected)
    {
        LVRInteractorStylePolygonPick::SetLassoTool();
    }else
    {
        LVRInteractorStylePolygonPick::SetPolygonTool();
    }
}

void LVRPickingInteractor::setLabel(int labelId, std::vector<int> pointIds)
{

    for (int pointId : pointIds)
    {
        m_pointLabels[pointId] = labelId;
        //TODO Check if other Actor was changed
    }
    //UpdateActor
    updateActor(labelId);
}

void LVRPickingInteractor::updateActor(int labelId)
{
    auto updateMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    auto updatedActor = vtkSmartPointer<vtkActor>::New();
    updatedActor->SetMapper(updateMapper);

    //Crete copy of points
    double point[3];
    auto selectedVtkPoints = vtkSmartPointer<vtkPoints>::New();
    int count = 0;
    for (int i = 0; i < m_pointLabels.size(); i++)
    {
        if (labelId == m_pointLabels[i])
        {
            count++;
            m_points->vtkDataSet::GetPoint(i, point);
            selectedVtkPoints->InsertNextPoint(point);
        }
    }
    Q_EMIT(pointsLabeled(labelId, count));

    auto updatedVtkPoly = vtkSmartPointer<vtkPolyData>::New();
    updatedVtkPoly->SetPoints(selectedVtkPoints);
    auto vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexFilter->SetInputData(updatedVtkPoly);
    vertexFilter->Update();

    auto polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->ShallowCopy(vertexFilter->GetOutput());


#if VTK_MAJOR_VERSION <= 5
    updatedMapper->SetInput(polyData);
#else
    updateMapper->SetInputData(polyData);
#endif
    updateMapper->ScalarVisibilityOff();   

    int r, g, b;
    m_labelColors[labelId].getRgb(&r, &g, &b);
    updatedActor->GetProperty()->SetColor(r/255.0, g/255.0, b/255.0); //(R,G,B)

    //Get Visiibility and Discard old Actor after that 
    bool visibility = true;
    if (m_labelActors.find(labelId) != m_labelActors.end())
    {
        visibility = m_labelActors[labelId]->GetVisibility();
        m_renderer->RemoveActor(m_labelActors[labelId]);
    }
    updatedActor->SetVisibility(visibility);
    m_labelActors[labelId] = updatedActor;
    m_renderer->AddActor(updatedActor);

    this->GetInteractor()->GetRenderWindow()->Render();
    this->HighlightProp(NULL);
}

void LVRPickingInteractor::labelModeChanged(bool setLabeling)
{

    //Check if no Labels were created
    if( m_labelColors.empty())
    {
        QMessageBox noLabelDialog;
        noLabelDialog.setText("No Label Instance was created! Create an instance beofre labeling Points.");
        noLabelDialog.setStandardButtons(QMessageBox::Ok);
        noLabelDialog.setIcon(QMessageBox::Warning);
        int returnValue = noLabelDialog.exec();
        return;

    }
    Q_EMIT(labelingStarted(true));
    m_pickMode = PickLabel;

}

std::vector<uint16_t>& LVRPickingInteractor::getLabeles()
{
    return m_pointLabels;
}
vtkSmartPointer<vtkPolyData> LVRPickingInteractor::getPoints()
{
    return m_points;
}
void LVRPickingInteractor::removeLabel(const int& id)
{
    if (m_labelActors.find(id) == m_labelActors.end())
    {
        //id unknown nothing to da
        return;
    }

    //Remove Actor
    m_renderer->RemoveActor(m_labelActors[id]);
    m_labelActors.erase(id);

    //update labeledpoints to unlabeled
    for(auto & pointlabel : m_pointLabels)
    {
        if (pointlabel == id)
        {
            //Set all points with this label to Unlabeled
            pointlabel = 0; 
        }
    }
    updateActor(0);

}

} /* namespace lvr2 */
