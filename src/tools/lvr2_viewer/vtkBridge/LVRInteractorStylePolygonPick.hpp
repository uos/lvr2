/**
 * @class   LVRInteractorStylePolygonPick
 * @brief   this can pick props underneath a rubber band selection
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
 */

#ifndef LVRInteractorStylePolygonPick_h
#define LVRInteractorStylePolygonPick_h

#include "vtkInteractionStyleModule.h"
#include "vtkInteractorStyleDrawPolygon.h"

#include "vtkVector.h"
#include <vector>

class VTKINTERACTIONSTYLE_EXPORT LVRInteractorStylePolygonPick : public vtkInteractorStyleDrawPolygon
{
public:
  static LVRInteractorStylePolygonPick* New();
  vtkTypeMacro(LVRInteractorStylePolygonPick, vtkInteractorStyleDrawPolygon);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void SetLassoTool();
  void SetPolygonTool();
  void StartSelect();
  void toggleSelectionMode();
  void resetSelection();
  int selectionPolygonSize();
  bool isPolygonToolSelected();

  void OnMouseMove() override;
  void OnLeftButtonDown() override;
  void OnLeftButtonUp() override;
  void OnChar() override;
  void OnKeyDown() override;

  std::vector<vtkVector2i> GetPolygonPoints();

protected:
  LVRInteractorStylePolygonPick();
  ~LVRInteractorStylePolygonPick() override;

  virtual void Pick();
 // virtual void DrawPolygon();

  bool firstPoint = true;

  int CurrentMode;
  bool lassoToolSelected = true;

private:
  LVRInteractorStylePolygonPick(const LVRInteractorStylePolygonPick&) = delete;
  void operator=(const LVRInteractorStylePolygonPick&) = delete;
  std::vector<vtkVector2i> polygonPoints;

};

#endif
