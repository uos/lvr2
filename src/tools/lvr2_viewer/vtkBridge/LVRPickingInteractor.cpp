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

namespace lvr2
{

LVRPickingInteractor::LVRPickingInteractor(vtkSmartPointer<vtkRenderer> renderer) :
        m_renderer(renderer)
{
    m_pickMode = None;
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
    m_renderer->AddActor(m_textActor);

}

LVRPickingInteractor::~LVRPickingInteractor()
{
    // TODO Auto-generated destructor stub
}


void LVRPickingInteractor::Dolly()
{
	vtkInteractorStyleTrackballCamera::Dolly();
}

void LVRPickingInteractor::Pan()
{
	vtkInteractorStyleTrackballCamera::Pan();
}

void LVRPickingInteractor::Spin()
{
	vtkInteractorStyleTrackballCamera::Spin();
}

void LVRPickingInteractor::Zoom()
{
	vtkInteractorStyleTrackballCamera::Zoom();
}

void LVRPickingInteractor::Rotate()
{
	vtkInteractorStyleTrackballCamera::Rotate();
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

void LVRPickingInteractor::OnLeftButtonDown()
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
    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

void LVRPickingInteractor::OnKeyPress()
{

}

void LVRPickingInteractor::OnKeyDown()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    std::string key = rwi->GetKeySym();

    if(key == "a" && m_correspondenceMode)
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

    if(key == "q")
    {
        m_textActor->VisibilityOff();
        m_pickMode = None;
        rwi->Render();
    }

}

void LVRPickingInteractor::OnKeyRelease()
{

}

} /* namespace lvr2 */
