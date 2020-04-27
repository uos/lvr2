/*=========================================================================

  Program:   Visualization Toolkit
  Module:    LVRInteractorStylePolygonPick.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   LVRInteractorStylePolygonPick
 * @brief   Like TrackBallCamera, but this can pick props underneath a rubber band selection
 * rectangle.
 *
 *
 * This interactor style allows the user to draw a rectangle in the render
 * window by hitting 'r' and then using the left mouse button.
 * When the mouse button is released, the attached picker operates on the pixel
 * in the center of the selection rectangle. If the picker happens to be a
 * vtkAreaPicker it will operate on the entire selection rectangle.
 * When the 'p' key is hit the above pick operation occurs on a 1x1 rectangle.
 * In other respects it behaves the same as its parent class.
 *
 * @sa
 * vtkAreaPicker
 */

#ifndef LVRInteractorStylePolygonPick_h
#define LVRInteractorStylePolygonPick_h

#include "vtkInteractionStyleModule.h" // For export macro
#include "vtkInteractorStyleTrackballCamera.h"

#include "vtkVector.h"
#include <vector>
class vtkUnsignedCharArray;

class VTKINTERACTIONSTYLE_EXPORT LVRInteractorStylePolygonPick
  : public vtkInteractorStyleTrackballCamera
{
public:
  static LVRInteractorStylePolygonPick* New();
  vtkTypeMacro(LVRInteractorStylePolygonPick, vtkInteractorStyleTrackballCamera);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void SetLassoTool();
  void SetPolygonTool();
  void StartSelect();

  //@{
  /**
   * Event bindings
   */
  void OnMouseMove() override;
  void OnLeftButtonDown() override;
  void OnLeftButtonUp() override;
  void OnChar() override;
  void OnKeyDown() override;
  //@}

  vtkSetMacro(DrawPolygonPixels, bool);
  vtkGetMacro(DrawPolygonPixels, bool);
  vtkBooleanMacro(DrawPolygonPixels, bool);

  std::vector<vtkVector2i> GetPolygonPoints();

protected:
  LVRInteractorStylePolygonPick();
  ~LVRInteractorStylePolygonPick() override;

  virtual void Pick();
  virtual void DrawPolygon();

  int StartPosition[2];
  int EndPosition[2];

  int Moving;
  bool firstPoint = true;

  vtkUnsignedCharArray* PixelArray;

  int CurrentMode;
  bool DrawPolygonPixels;
  bool lassoToolSelected = true;

private:
  LVRInteractorStylePolygonPick(const LVRInteractorStylePolygonPick&) = delete;
  void operator=(const LVRInteractorStylePolygonPick&) = delete;

  class vtkInternal;
  vtkInternal* Internal;
};

#endif
