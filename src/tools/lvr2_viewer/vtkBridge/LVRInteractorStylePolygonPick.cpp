#include "LVRInteractorStylePolygonPick.hpp"
#include <vtkAbstractPropPicker.h>
#include <vtkNew.h>
#include <vtkAreaPicker.h>
#include <vtkAssemblyPath.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVectorOperators.h>
#include <vtkCommand.h>
#include <algorithm>

vtkStandardNewMacro(LVRInteractorStylePolygonPick);

#define VTKISRBP_ORIENT 0
#define VTKISRBP_SELECT 1

LVRInteractorStylePolygonPick::LVRInteractorStylePolygonPick()
{
    this->CurrentMode = VTKISRBP_ORIENT;
}

LVRInteractorStylePolygonPick::~LVRInteractorStylePolygonPick()
{

}

void LVRInteractorStylePolygonPick::StartSelect()
{
    this->CurrentMode = VTKISRBP_SELECT;
}

std::vector<vtkVector2i> LVRInteractorStylePolygonPick::GetPolygonPoints()
{
    if (lassoToolSelected)
    {
        return vtkInteractorStyleDrawPolygon::GetPolygonPoints();
    } else
    {
        return polygonPoints;
    }
}

void LVRInteractorStylePolygonPick::OnChar()
{
  switch (this->Interactor->GetKeyCode())
  {
    case 'l':
    case 'L':
      // l toggles the rubber band selection mode for mouse button 1
      //toggleSelectionMode();
      break;
    case 'p':
    case 'P':
    {
        /*
      vtkRenderWindowInteractor* rwi = this->Interactor;
      int* eventPos = rwi->GetEventPosition();
      this->FindPokedRenderer(eventPos[0], eventPos[1]);
      this->StartPosition[0] = eventPos[0];
      this->StartPosition[1] = eventPos[1];
      this->EndPosition[0] = eventPos[0];
      this->EndPosition[1] = eventPos[1];
      this->Pick();
      break;*/
    }
    default:
      this->Superclass::OnChar();
  }
}

void LVRInteractorStylePolygonPick::OnLeftButtonDown()
{
  
    if (this->CurrentMode != VTKISRBP_SELECT)
    {
        // if not in rubber band mode, let the parent class handle it
        this->Superclass::OnLeftButtonDown();
        return;
    }
    if (lassoToolSelected)
    {
        //Lasso Tool
        vtkInteractorStyleDrawPolygon::OnLeftButtonDown();
        this->FindPokedRenderer(10,10);
    } else 
    {
        //Polygon Tool
      //  vtkVector2i newPoint(this->Interactor->GetEventPosition()[0],
      //                          this->Interactor->GetEventPosition()[1]);
        std::cout << this->Interactor->GetEventPosition()[0] << std::endl;
        std::cout << this->Interactor->GetEventPosition()[1] << std::endl;
        std::cout << this->Interactor->GetEventPosition()[2] << std::endl;

    }

}

void LVRInteractorStylePolygonPick::OnMouseMove()
{
    if (this->CurrentMode != VTKISRBP_SELECT)
    {
        // if not in rubber band mode,  let the parent class handle it
        this->Superclass::OnMouseMove();
        return;
    }
    if (lassoToolSelected)
    {
        //Lasso Tool
        vtkInteractorStyleDrawPolygon::OnMouseMove();
    } else 
    {
        //Polygon Tool
    }
}

void LVRInteractorStylePolygonPick::OnLeftButtonUp()
{
    if (this->CurrentMode != VTKISRBP_SELECT)
    {
        // if not in rubber band mode,  let the parent class handle it
        this->Superclass::OnLeftButtonUp();
        return;
    }
    if (lassoToolSelected)
    {
        //Lasso Tool
        vtkInteractorStyleDrawPolygon::OnLeftButtonUp();
    } else
    {
        //Polygon Tool
        polygonPoints.push_back(vtkVector2i(this->Interactor->GetEventPosition()[0],
                                this->Interactor->GetEventPosition()[1]));
    }

    this->FindPokedRenderer(10,10);

    if (this->selectionPolygonSize() >= 3)
    {
        this->Pick();
    }
}



inline bool compareX(vtkVector2i i, vtkVector2i j) {return (i[0] < j[0]);};
inline bool compareY(vtkVector2i i, vtkVector2i j) {return (i[1] < j[1]);};
void LVRInteractorStylePolygonPick::Pick()
{
  // calculate binding box
  std::vector<vtkVector2i> polygonPoints = this->GetPolygonPoints();
  double rbcenter[3];
  int* size = this->Interactor->GetRenderWindow()->GetSize();
  int min[2], max[2];
  
  min[0] = std::min_element(polygonPoints.begin(), polygonPoints.end(), compareX)->GetX();
  min[1] = std::min_element(polygonPoints.begin(), polygonPoints.end(), compareY)->GetY();

  max[0] = std::max_element(polygonPoints.begin(), polygonPoints.end(), compareX)->GetX();
  max[1] = std::max_element(polygonPoints.begin(), polygonPoints.end(), compareY)->GetY();

  rbcenter[0] = (min[0] + max[0]) / 2.0;
  rbcenter[1] = (min[1] + max[1]) / 2.0;
  rbcenter[2] = 0;

  if (this->State == VTKIS_NONE)
  {
    // tell the RenderWindowInteractor's picker to make it happen
    vtkRenderWindowInteractor* rwi = this->Interactor;

    vtkAssemblyPath* path = nullptr;
    rwi->StartPickCallback();
    vtkAbstractPropPicker* picker = vtkAbstractPropPicker::SafeDownCast(rwi->GetPicker());
    if (picker != nullptr)
    {
      vtkAreaPicker* areaPicker = vtkAreaPicker::SafeDownCast(picker);
      if (areaPicker != nullptr)
      {
        areaPicker->AreaPick(min[0], min[1], max[0], max[1], this->CurrentRenderer);
      }
      else
      {
        picker->Pick(rbcenter[0], rbcenter[1], 0.0, this->CurrentRenderer);
      }
      path = picker->GetPath();
    }
    if (path == nullptr)
    {
      this->HighlightProp(nullptr);
      this->PropPicked = 0;
    }
    else
    {
      // highlight the one prop that the picker saved in the path
      // this->HighlightProp(path->GetFirstNode()->GetViewProp());
      this->PropPicked = 1;
    }
    rwi->EndPickCallback();
  }

  this->Interactor->Render();
}
/*
//--------------------------------------------------------------------------
void LVRInteractorStylePolygonPick::DrawPolygon()
{
  vtkNew<vtkUnsignedCharArray> tmpPixelArray;
  tmpPixelArray->DeepCopy(this->PixelArray);
  unsigned char* pixels = tmpPixelArray->GetPointer(0);
  int* size = this->Interactor->GetRenderWindow()->GetSize();

  // draw each line segment
  for (vtkIdType i = 0; i < this->Internal->GetNumberOfPoints() - 1; i++)
  {
    const vtkVector2i& a = this->Internal->GetPoint(i);
    const vtkVector2i& b = this->Internal->GetPoint(i + 1);

    this->Internal->DrawPixels(a, b, pixels, size);
  }

  // draw a line from the end to the start
  if (this->Internal->GetNumberOfPoints() >= 3)
  {
    const vtkVector2i& start = this->Internal->GetPoint(0);
    const vtkVector2i& end = this->Internal->GetPoint(this->Internal->GetNumberOfPoints() - 1);

    this->Internal->DrawPixels(start, end, pixels, size);
  }

  this->Interactor->GetRenderWindow()->SetPixelData(0, 0, size[0] - 1, size[1] - 1, pixels, 0);
  this->Interactor->GetRenderWindow()->Frame();
}
*/
//--------------------------------------------------------------------------
void LVRInteractorStylePolygonPick::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void LVRInteractorStylePolygonPick::SetPolygonTool()
{
  lassoToolSelected = false; 
}

void LVRInteractorStylePolygonPick::SetLassoTool()
{
  lassoToolSelected = true; 
}
void LVRInteractorStylePolygonPick::OnKeyDown()
{
    if (!lassoToolSelected)// && "Return" == this->Interactor->GetKeySym())
    {
        if (this->CurrentMode != VTKISRBP_SELECT)
          {
            // if not in rubber band mode,  let the parent class handle it
        //    this->Superclass::OnKeyDown();
       //     return;
          }

        // otherwise record the rubber band end coordinate and then fire off a pick
        if ((this->StartPosition[0] != this->EndPosition[0]) ||
            (this->StartPosition[1] != this->EndPosition[1]))
        {
            this->Pick();
            firstPoint = true;
        }
        this->Moving = 0;
    }
}
void LVRInteractorStylePolygonPick::toggleSelectionMode()
{
      if (this->CurrentMode == VTKISRBP_ORIENT)
      {
        this->CurrentMode = VTKISRBP_SELECT;
      }
      else
      {
        this->CurrentMode = VTKISRBP_ORIENT;
      }
}

bool LVRInteractorStylePolygonPick::isPolygonToolSelected()
{
    return !lassoToolSelected;
}
int LVRInteractorStylePolygonPick::selectionPolygonSize()
{
    if (lassoToolSelected)
    {
        return vtkInteractorStyleDrawPolygon::GetPolygonPoints().size();
    } else
    {
        return polygonPoints.size();
    }
}
void LVRInteractorStylePolygonPick::resetSelection()
{
    polygonPoints.clear();
}
