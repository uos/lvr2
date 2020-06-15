/*=========================================================================

  Program:   Visualization Toolkit
  Module:    LVRInteractorStylePolygonPick.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
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

//-----------------------------------------------------------------------------
class LVRInteractorStylePolygonPick::vtkInternal
{
public:
  std::vector<vtkVector2i> points;

  void PopPoint()
  {
    this->points.pop_back();
  }

  void AddPoint(const vtkVector2i& point) { this->points.push_back(point); }

  void AddPoint(int x, int y) { this->AddPoint(vtkVector2i(x, y)); }

  void RemoveLastPoint(){this->points.pop_back();}
  vtkVector2i GetPoint(vtkIdType index) const { return this->points[index]; }

  vtkIdType GetNumberOfPoints() const { return static_cast<vtkIdType>(this->points.size()); }

  void Clear() { this->points.clear(); }

  void DrawPixels(
    const vtkVector2i& StartPos, const vtkVector2i& EndPos, unsigned char* pixels, int* size)
  {
    int x1 = StartPos.GetX(), x2 = EndPos.GetX();
    int y1 = StartPos.GetY(), y2 = EndPos.GetY();

    double x = x2 - x1;
    double y = y2 - y1;
    double length = sqrt(x * x + y * y);
    if (length == 0)
    {
      return;
    }
    double addx = x / length;
    double addy = y / length;

    x = x1;
    y = y1;
    int row, col;
    for (double i = 0; i < length; i += 1)
    {
      col = (int)x;
      row = (int)y;
      pixels[3 * (row * size[0] + col)] = 255 ^ pixels[3 * (row * size[0] + col)];
      pixels[3 * (row * size[0] + col) + 1] = 255 ^ pixels[3 * (row * size[0] + col) + 1];
      pixels[3 * (row * size[0] + col) + 2] = 255 ^ pixels[3 * (row * size[0] + col) + 2];
      x += addx;
      y += addy;
    }
  }
};

//--------------------------------------------------------------------------
LVRInteractorStylePolygonPick::LVRInteractorStylePolygonPick()
{
  this->Internal = new vtkInternal();
  this->CurrentMode = VTKISRBP_ORIENT;
  this->StartPosition[0] = this->StartPosition[1] = 0;
  this->EndPosition[0] = this->EndPosition[1] = 0;
  this->Moving = 0;
  this->DrawPolygonPixels = true;
  this->PixelArray = vtkUnsignedCharArray::New();
}

//--------------------------------------------------------------------------
LVRInteractorStylePolygonPick::~LVRInteractorStylePolygonPick()
{
  this->PixelArray->Delete();
  delete this->Internal;
}

//--------------------------------------------------------------------------
void LVRInteractorStylePolygonPick::StartSelect()
{
  this->CurrentMode = VTKISRBP_SELECT;
}

//----------------------------------------------------------------------------
std::vector<vtkVector2i> LVRInteractorStylePolygonPick::GetPolygonPoints()
{
  return this->Internal->points;
}

//--------------------------------------------------------------------------
void LVRInteractorStylePolygonPick::OnChar()
{
  switch (this->Interactor->GetKeyCode())
  {
    case 'l':
    case 'L':
      // r toggles the rubber band selection mode for mouse button 1
      if (this->CurrentMode == VTKISRBP_ORIENT)
      {
        this->CurrentMode = VTKISRBP_SELECT;
      }
      else
      {
        this->CurrentMode = VTKISRBP_ORIENT;
      }
      break;
    case 'p':
    case 'P':
    {
      vtkRenderWindowInteractor* rwi = this->Interactor;
      int* eventPos = rwi->GetEventPosition();
      this->FindPokedRenderer(eventPos[0], eventPos[1]);
      this->StartPosition[0] = eventPos[0];
      this->StartPosition[1] = eventPos[1];
      this->EndPosition[0] = eventPos[0];
      this->EndPosition[1] = eventPos[1];
      this->Pick();
      break;
    }
    default:
      this->Superclass::OnChar();
  }
}

//--------------------------------------------------------------------------
void LVRInteractorStylePolygonPick::OnLeftButtonDown()
{
  
  if (this->CurrentMode != VTKISRBP_SELECT)
  {
    // if not in rubber band mode, let the parent class handle it
    this->Superclass::OnLeftButtonDown();
    return;
  }

  if (!this->Interactor)
  {
    return;
  }

  // otherwise record the rubber band starting coordinate

  this->Moving = 1;
  if (!lassoToolSelected)
  {
  	if (!firstPoint)
  	{
  	  return;
  	}
  	firstPoint = false;
  }


  vtkRenderWindow* renWin = this->Interactor->GetRenderWindow();

  this->StartPosition[0] = this->Interactor->GetEventPosition()[0];
  this->StartPosition[1] = this->Interactor->GetEventPosition()[1];
  this->EndPosition[0] = this->StartPosition[0];
  this->EndPosition[1] = this->StartPosition[1];

  this->PixelArray->Initialize();
  this->PixelArray->SetNumberOfComponents(3);
  int* size = renWin->GetSize();
  this->PixelArray->SetNumberOfTuples(size[0] * size[1]);

  renWin->GetPixelData(0, 0, size[0] - 1, size[1] - 1, 1, this->PixelArray);
  this->Internal->Clear();
  this->Internal->AddPoint(this->StartPosition[0], this->StartPosition[1]);
  this->InvokeEvent(vtkCommand::StartInteractionEvent);

  //renWin->GetRGBACharPixelData(0, 0, size[0] - 1, size[1] - 1, 1, this->PixelArray);
  this->FindPokedRenderer(this->StartPosition[0], this->StartPosition[1]);
}

//--------------------------------------------------------------------------
void LVRInteractorStylePolygonPick::OnMouseMove()
{
  if (this->CurrentMode != VTKISRBP_SELECT)
  {
    // if not in rubber band mode,  let the parent class handle it
    this->Superclass::OnMouseMove();
    return;
  }

  if (!this->Interactor || !this->Moving)
  {
    return;
  }

  this->EndPosition[0] = this->Interactor->GetEventPosition()[0];
  this->EndPosition[1] = this->Interactor->GetEventPosition()[1];
  int* size = this->Interactor->GetRenderWindow()->GetSize();
  if (this->EndPosition[0] > (size[0] - 1))
  {
    this->EndPosition[0] = size[0] - 1;
  }
  if (this->EndPosition[0] < 0)
  {
    this->EndPosition[0] = 0;
  }
  if (this->EndPosition[1] > (size[1] - 1))
  {
    this->EndPosition[1] = size[1] - 1;
  }
  if (this->EndPosition[1] < 0)
  {
    this->EndPosition[1] = 0;
  }

  vtkVector2i lastPoint = this->Internal->GetPoint(this->Internal->GetNumberOfPoints() - 1);
  vtkVector2i newPoint(this->EndPosition[0], this->EndPosition[1]);
  if ((lastPoint - newPoint).SquaredNorm() > 100)
  {
    if (!lassoToolSelected && this->Internal->GetNumberOfPoints() > 2)
    {
	    this->Internal->RemoveLastPoint();
    }
    this->Internal->AddPoint(newPoint);
    if (this->DrawPolygonPixels)
    {
      this->DrawPolygon();
    }
  }
}

//--------------------------------------------------------------------------
void LVRInteractorStylePolygonPick::OnLeftButtonUp()
{
  if (this->CurrentMode != VTKISRBP_SELECT)
  {
    // if not in rubber band mode,  let the parent class handle it
    this->Superclass::OnLeftButtonUp();
    return;
  }

  if (!this->Interactor || !this->Moving)
  {
    return;
  }

  if (!lassoToolSelected)
  {
    
    vtkVector2i newPoint(this->Interactor->GetEventPosition()[0], this->Interactor->GetEventPosition()[1]);
    this->Internal->AddPoint(newPoint);
    if (this->DrawPolygonPixels)
    {
      this->DrawPolygon();
    }
    this->Moving = 0;
    return;
  }

  // otherwise record the rubber band end coordinate and then fire off a pick
  if ((this->StartPosition[0] != this->EndPosition[0]) ||
    (this->StartPosition[1] != this->EndPosition[1]))
  {
    this->Pick();
  }
  this->Moving = 0;
  // this->CurrentMode = VTKISRBP_ORIENT;
}



inline bool compareX(vtkVector2i i, vtkVector2i j) {return (i[0] < j[0]);};
inline bool compareY(vtkVector2i i, vtkVector2i j) {return (i[1] < j[1]);};
//--------------------------------------------------------------------------
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
  //std::cout << "Minimum : " << min[0] << "  " << min[1] << std::endl;
  //std::cout << "Maximum : " << max[0] << "  " << max[1] << std::endl;
  //std::cout << "Count: " << polygonPoints.size() << std::endl;

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

//--------------------------------------------------------------------------
void LVRInteractorStylePolygonPick::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void LVRInteractorStylePolygonPick::SetPolygonTool()
{
    std::cout << "lasso" << std::endl;
  lassoToolSelected = false; 
}

void LVRInteractorStylePolygonPick::SetLassoTool()
{
    std::cout << "no lasso" << std::endl;

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
            std::cout << "did this " << std::endl;
            this->Pick();
            firstPoint = true;
        }
        this->Moving = 0;
    }
}
