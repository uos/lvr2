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
    m_pixel = vtkUnsignedCharArray::New();
    
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
    this->Superclass::OnChar();
}
void LVRInteractorStylePolygonPick::createPixelArray()
{
    vtkNew<vtkUnsignedCharArray> tmpPixelArray;
    const int* windowSize = this->Interactor->GetRenderWindow()->GetSize();
    tmpPixelArray->DeepCopy(m_pixel);
    unsigned char* pixels = tmpPixelArray->GetPointer(0);
    if(moving)
    {
        polygonPoints.push_back(movingPoint);
    }
    for(size_t i = 0; i < polygonPoints.size(); i++)
    {
        if(i < polygonPoints.size() - 1)
        {
            createPixelLine(polygonPoints[i], polygonPoints[i + 1], pixels, windowSize); 
        }
        else if(polygonPoints.size() > 2)
        {
            //Last Point connect First and Last Element ignore if only a line
            createPixelLine(polygonPoints[i], polygonPoints[0], pixels, windowSize); 
        }
    }
    this->Interactor->GetRenderWindow()->SetPixelData(0, 0, windowSize[0] - 1, windowSize[1] - 1, pixels, 1);
    if(moving)
    {
        polygonPoints.pop_back();
    }
    //this->Interactor->GetRenderWindow()->Frame();
}

//Copied From VTK InteractorStyleDrawPolygon 9.0.20210107 
void LVRInteractorStylePolygonPick::createPixelLine(const vtkVector2i& StartPos, const vtkVector2i& EndPos, unsigned char* pixels, const int* size)
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
        if(!moving)
        {
            moving = true;
            const int* windowSize = this->Interactor->GetRenderWindow()->GetSize();
            this->Interactor->GetRenderWindow()->GetPixelData(0, 0, windowSize[0] - 1, windowSize[1] - 1, 1, m_pixel);
        }
        //Polygon Tool
        if (polygonPoints.empty())
        {
            polygonPoints.push_back(vtkVector2i(this->Interactor->GetEventPosition()[0],
                                this->Interactor->GetEventPosition()[1]));
        }
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
        if(moving)
        {
            movingPoint = vtkVector2i(this->Interactor->GetEventPosition()[0],
                                this->Interactor->GetEventPosition()[1]);
        }
        createPixelArray();
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
        //moving = false;
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
            this->PropPicked = 1;
        }
        rwi->EndPickCallback();
    }
    this->Interactor->Render();
}

void LVRInteractorStylePolygonPick::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}

void LVRInteractorStylePolygonPick::SetPolygonTool()
{
    lassoToolSelected = false; 
    const int* size = this->Interactor->GetRenderWindow()->GetSize();
    m_pixel->Initialize();
    m_pixel->SetNumberOfComponents(3);
    m_pixel->SetNumberOfTuples(size[0] * size[1]);
    this->Interactor->GetRenderWindow()->GetPixelData(0, 0, size[0] - 1, size[1] - 1, 1, m_pixel);
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
    moving = false;
    polygonPoints.clear();
    lassoToolSelected = false; 
    const int* size = this->Interactor->GetRenderWindow()->GetSize();
    m_pixel->Initialize();
    m_pixel->SetNumberOfComponents(3);
    m_pixel->SetNumberOfTuples(size[0] * size[1]);
    this->Interactor->GetRenderWindow()->GetPixelData(0, 0, size[0] - 1, size[1] - 1, 1, m_pixel);
}
