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
 * MainWindow.cpp
 *
 *  @date Jan 31, 2014
 *  @author Thomas Wiemann
 */

#include "LVRTimerCallback.hpp"

namespace lvr
{
    LVRTimerCallback::LVRTimerCallback()
    {
        m_firstrun = true;
    }

    LVRTimerCallback::~LVRTimerCallback()
    {
    }

    LVRTimerCallback* LVRTimerCallback::New()
    {
        LVRTimerCallback* cb = new LVRTimerCallback;
        return cb;
    }

    void LVRTimerCallback::setPathCamera(vtkSmartPointer<vtkCameraRepresentation> pathCamera)
    {
        m_pathCamera = pathCamera;
    }

    void LVRTimerCallback::Execute(vtkObject* caller, unsigned long eventId, void* callData)
    {
        if(vtkCommand::TimerEvent == eventId)
        {
            if(m_firstrun)
            {
                m_firstrun = false;
                m_pathCamera->InitializePath();
            }
            m_pathCamera->AddCameraToPath();
        }
    }
} /* namespace lvr */
