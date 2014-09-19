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
 * LVRPickingInteractor.hpp
 *
 *  @date Feb 19, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRPICKINGINTERACTOR_HPP_
#define LVRPICKINGINTERACTOR_HPP_

#include <QObject>

#include <vtkTextActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

namespace lvr
{

enum PickMode {None, PickPoint, PickFirst, PickSecond};

class LVRPickingInteractor : public QObject, public vtkInteractorStyleTrackballCamera
{
    Q_OBJECT
public:
    LVRPickingInteractor(vtkSmartPointer<vtkRenderer> renderer);
    virtual ~LVRPickingInteractor();

    /**
     * @brief   Overloaded mouse event handling.
     */
    virtual void OnLeftButtonDown();

    /**
     * @brief   Overloaded keyboard press event handling
     */
    virtual void OnKeyPress();

    /**
     * @brief   Overloaded keyboard release event handling
     */
    virtual void OnKeyRelease();

    virtual void OnKeyDown();

    virtual void OnTimer();
    virtual void OnExpose();
    virtual void OnConfigure();

    /**
     * @brief   returns the text-actor, needed to readd-it after clearing the render window
     */
    vtkSmartPointer<vtkTextActor>   getTextActor(){ return m_textActor; }

public Q_SLOTS:
    void correspondenceSearchOn();
    void correspondenceSearchOff();

Q_SIGNALS:
    void firstPointPicked(double*);
    void secondPointPicked(double*);

private:
    /// Indicates picking mode
    PickMode            m_pickMode;

    /// Text actor to display info if in picking mode
    vtkSmartPointer<vtkTextActor>   m_textActor;

    vtkSmartPointer<vtkRenderer>    m_renderer;

    bool                            m_correspondenceMode;

};

} /* namespace lvr */

#endif /* LVRPICKINGINTERACTOR_HPP_ */
