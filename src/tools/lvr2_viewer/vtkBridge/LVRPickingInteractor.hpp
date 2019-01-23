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
 * LVRPickingInteractor.hpp
 *
 *  @date Feb 19, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRPICKINGINTERACTOR_HPP_
#define LVRPICKINGINTERACTOR_HPP_

#include <QObject>
#include <QMessageBox>

#include <vtkTextActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

namespace lvr2
{


class LVRPickingInteractor : public QObject, public vtkInteractorStyle
{
    Q_OBJECT
public:
    LVRPickingInteractor(vtkSmartPointer<vtkRenderer> renderer);
    virtual ~LVRPickingInteractor();

    /**
     * @brief   Overloaded mouse event handling.
     */
    virtual void OnLeftButtonDown();
    virtual void OnLeftButtonUp();
    virtual void OnMouseMove();
    virtual void OnMiddleButtonUp();
    virtual void OnMiddleButtonDown();
    virtual void OnRightButtonUp();
    virtual void OnRightButtonDown();
    virtual void OnMouseWheelBackward();
    virtual void OnMouseWheelForward();

    virtual void OnChar();
    virtual void OnTimer();

    /**
     * @brief   Overloaded keyboard press event handling
     */
    virtual void OnKeyPress();

    /**
     * @brief   Overloaded keyboard release event handling
     */
    virtual void OnKeyRelease();

    virtual void OnKeyDown();

    virtual void Dolly();
    virtual void Dolly(double factor);
    virtual void Pan();
    virtual void Spin();
    virtual void Zoom();
    virtual void Rotate();


    /**
     * @brief   returns the text-actor, needed to readd-it after clearing the render window
     */
    vtkSmartPointer<vtkTextActor>   getTextActor(){ return m_textActor; }

    void updateFocalPoint();

public Q_SLOTS:
    void correspondenceSearchOn();
    void correspondenceSearchOff();

    void setMotionFactor(double factor);
    void setRotationFactor(double factor);

    void setFocalPointRendering(int state);
    void setStereoMode(int state);

    void pickFocalPoint();

    void modeTerrain();
    void modeTrackball();
    void modeShooter();



Q_SIGNALS:
    void firstPointPicked(double*);
    void secondPointPicked(double*);
    void pointSelected(vtkActor*, int);

private:

    enum InteractorMode {TRACKBALL, SHOOTER, TERRAIN};
    enum ShooterMode {LOOK, HOVER};
    enum PickMode {None, PickPoint, PickFirst, PickSecond, PickFocal};

    void handlePicking();

    // ------------------------- TRACKBALL

    void dollyTrackball();
    void dollyTrackball(double factor);
    void panTrackball();
    void spinTrackball();
    void zoomTrackball();
    void rotateTrackball();

    void onLeftButtonDownTrackball();
    void onLeftButtonUpTrackball();
    void onMouseMoveTrackball();
    void onMiddleButtonUpTrackball();
    void onMiddleButtonDownTrackball();
    void onRightButtonUpTrackball();
    void onRightButtonDownTrackball();
    void onMouseWheelBackwardTrackball();
    void onMouseWheelForwardTrackball();

    // ------------------------ TERRAIN

    void dollyTerrain();
    void dollyTerrain(double factor);
    void panTerrain();
    void spinTerrain();
    void zoomTerrain();
    void rotateTerrain();

    void onLeftButtonDownTerrain();
    void onLeftButtonUpTerrain();
    void onMouseMoveTerrain();
    void onMiddleButtonUpTerrain();
    void onMiddleButtonDownTerrain();
    void onRightButtonUpTerrain();
    void onRightButtonDownTerrain();
    void onMouseWheelBackwardTerrain();
    void onMouseWheelForwardTerrain();

    // ------------------------ SHOOTER

    void dollyShooter();
    void dollyShooter(double factor);
    void panShooter();
    void spinShooter();
    void zoomShooter();
    void rotateShooter();
    void hoverShooter();
    void resetViewUpShooter();
    void strafeShooter(double factor);

    void onLeftButtonDownShooter();
    void onLeftButtonUpShooter();
    void onMouseMoveShooter();
    void onMiddleButtonUpShooter();
    void onMiddleButtonDownShooter();
    void onRightButtonUpShooter();
    void onRightButtonDownShooter();
    void onMouseWheelBackwardShooter();
    void onMouseWheelForwardShooter();





    /// Indicates picking mode
    PickMode            m_pickMode;

    /// Text actor to display info if in picking mode
    vtkSmartPointer<vtkTextActor>   m_textActor;
    vtkSmartPointer<vtkActor>       m_sphereActor;

    vtkSmartPointer<vtkRenderer>    m_renderer;

    bool                            m_correspondenceMode;

    unsigned int                    m_numberOfClicks;
    int                             m_previousPosition[2];
    int                             m_startCameraMovePosition[2];

    double                          m_viewUp[3];

    float                           m_motionFactor;
    float                           m_rotationFactor;

    InteractorMode                  m_interactorMode;
    ShooterMode                     m_shooterMode;


};

} /* namespace lvr2 */

#endif /* LVRPICKINGINTERACTOR_HPP_ */
