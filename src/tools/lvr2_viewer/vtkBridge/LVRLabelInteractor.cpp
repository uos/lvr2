#include "LVRLabelInteractor.hpp"
#include <vtkCoordinate.h>
#include <QInputDialog>
// Define interaction style
//

    
vtkStandardNewMacro(LVRLabelInteractorStyle);

LVRLabelInteractorStyle::LVRLabelInteractorStyle()
{
      m_selectedIds = vtkSmartPointer<vtkIdTypeArray>::New(); 
      SelectedMapper = vtkSmartPointer<vtkDataSetMapper>::New();
      SelectedActor = vtkSmartPointer<vtkActor>::New();
      m_labelList << QString("NoLabel");
      //m_SelectedPoints = vtkSmartPointer<vtkPolyData>::New();
      SelectedActor->SetMapper(SelectedMapper);
      colors = {}; 
      colors.push_back({255, 255, 102});
      colors.push_back({127, 0, 255});
      colors.push_back({255,0,255});
      colors.push_back({0,255 ,0 });
      colors.push_back({255, 0, 127});
      colors.push_back({0,128 ,255});
      colors.push_back({0,204 ,102});
      colors.push_back({153, 204, 255});
      colors.push_back({153, 255, 204});
      colors.push_back({255, 153, 153});
      colors.push_back({153, 76, 0});
      colors.push_back({0, 255, 128});
}
void LVRLabelInteractorStyle::OnRightButtonDown()
{

	LVRInteractorStylePolygonPick::OnLeftButtonDown();
}

void LVRLabelInteractorStyle::OnRightButtonUp()
{
	calculateSelection(false);
}
void LVRLabelInteractorStyle::OnLeftButtonUp()
{
	calculateSelection(true);
}
bool isInside(std::vector<vtkVector2i> polygon, int& pX, int& pY)
{
	int n = polygon.size();
	if (n < 3) return false;

	int i, j, c = 0;
	for(i = 0, j = n -1; i < n ; j = i++)
	{
		auto& vert = polygon[j];
		auto& vert_next = polygon[i];

		if ( ((vert_next.GetY()>pY) != (vert.GetY()>pY)) &&
			(pX < (vert.GetX()-vert_next.GetX()) * (pY-vert_next.GetY()) / (vert.GetY()-vert_next.GetY()) + vert_next.GetX())   )
		{
			c = !c;
		}
	}
            return c;
}


void LVRLabelInteractorStyle::calculateSelection(bool select)
{

      // Forward events
      LVRInteractorStylePolygonPick::OnLeftButtonUp();

      if (!m_points)
      {
	      return;
      }

      this->CurrentRenderer->RemoveActor(SelectedActor);
      SelectedActor = vtkSmartPointer<vtkActor>::New();
      SelectedMapper = vtkSmartPointer<vtkDataSetMapper>::New();
      SelectedActor->SetMapper(SelectedMapper);
  

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
	if (isInside(polygonPoints, displayCoord[0], displayCoord[1]))
	{
		selectedPolyPoints.push_back(ids->GetValue(i));
	}
      }

#if VTK_MAJOR_VERSION <= 5
      SelectedMapper->SetInput(selected);
#else
      SelectedMapper->SetInputData(selected);
#endif
      SelectedMapper->ScalarVisibilityOff();
/*
      for(vtkIdType i = 0; i < ids->GetNumberOfTuples(); i++)
      */

      for(auto selectedPolyPoint : selectedPolyPoints)
      {
		m_SelectedPoints[selectedPolyPoint] = select;
      }

      auto vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
      auto foo = vtkSmartPointer<vtkPoints>::New();
      auto bar = vtkSmartPointer<vtkPolyData>::New();
      double point[3];


      for(int i = 0; i < m_SelectedPoints.size(); i++)
      {
	      if(m_SelectedPoints[i])
	      {
      		m_points->vtkDataSet::GetPoint(i,point);
      		foo->InsertNextPoint(point);
	      }
      }
      	bar->SetPoints(foo);


      vertexFilter->SetInputData(bar);
      vertexFilter->Update();

      auto polyData = vtkSmartPointer<vtkPolyData>::New();
      polyData->ShallowCopy(vertexFilter->GetOutput());

#if VTK_MAJOR_VERSION <= 5
       SelectedMapper->SetInput(polyData);
#else
      SelectedMapper->SetInputData(polyData);
#endif
      SelectedMapper->ScalarVisibilityOff();   


      SelectedActor->GetProperty()->SetColor(1.0, 0.0, 0.0); //(R,G,B)
      //SelectedActor->GetProperty()->SetPointSize(3);

      this->CurrentRenderer->AddActor(SelectedActor);
      this->GetInteractor()->GetRenderWindow()->Render();
      this->HighlightProp(NULL);
}

void LVRLabelInteractorStyle::labelSelectedPoints(QString label)
{

}

void LVRLabelInteractorStyle::extractLabel()
{
	ofstream outfile("Test.txt");
	for (int i = 0; i < m_pointLabels.size(); i++)
	{
		if (m_pointLabels[i] != 0)
		{
			outfile << i << ":" << unsigned(m_pointLabels[i]) << "\n";
		}

	}
	outfile.close();

}
void LVRLabelInteractorStyle::OnKeyUp()
{
    vtkRenderWindowInteractor *rwi = this->Interactor;
    std::string key = rwi->GetKeySym();
    if (key == "Left")
    {
	if (!m_points)
	{
		return;
	}

   	bool accepted;
   	QString label = QInputDialog::getItem(0, "Select Label", "Choose Label For SelectedPoints. You can press n to add a new Label:",m_labelList , 0, true, &accepted);
	bool updateActors = false;
	if (accepted)
	{
		uint8_t labelIndex =  m_labelList.indexOf(label);
      		for(vtkIdType i = 0; i < m_selectedIds->GetNumberOfTuples(); i++)
      		{
			auto id = m_selectedIds->GetValue(i);
			//Check if actors need to updated
			if(labelIndex != 0 && m_pointLabels[id] != 0 && m_pointLabels[id] != labelIndex)
			{
				updateActors = true;
			} else if(labelIndex == 0 && (m_pointLabels[id] != 0))
			{

				updateActors = true;
			}
			m_pointLabels[id] = labelIndex;
		}
		if (updateActors)
		{
		}
		foo2[labelIndex - 1] = m_SelectedPoints;
		

		m_SelectedPoints = std::vector<bool>(m_points->GetNumberOfPoints(), false);
      		auto newActor = vtkSmartPointer<vtkActor>::New();
		newActor = SelectedActor;
      		newActor->GetProperty()->SetColor(colors[labelIndex - 1][0] /255.0,colors[labelIndex - 1][1] / 255.0,colors[labelIndex - 1][2] / 255.0); //(R,G,B)
		this->CurrentRenderer->RemoveActor(m_labelActors[labelIndex - 1]);
		m_labelActors[labelIndex - 1] = newActor;
		this->CurrentRenderer->RemoveActor(SelectedActor);
      		SelectedActor = vtkSmartPointer<vtkActor>::New();
	        this->CurrentRenderer->AddActor(newActor);
      		this->GetInteractor()->GetRenderWindow()->Render();
      		this->HighlightProp(NULL);

      	}

		
    }
    if (key == "m")
    {
   	bool accepted;
   	QString label = QInputDialog::getItem(0, "Modify Selected Points", "Choose Label which should be modified:",m_labelList , 0, true, &accepted);
	if (accepted)
	{
	    m_SelectedPoints = foo2[m_labelList.indexOf(label) - 1];

	}


    }
    if (key == "n")
    {
	bool accepted;

	QString text = QInputDialog::getText(0, "Insert New Label", "New Label:", QLineEdit::Normal, "", &accepted);
	if (accepted)
	{
		if(m_labelList.indexOf(text) != -1)
		{
				return;
		}
		foo2.push_back(std::vector<bool>(m_points->GetNumberOfPoints(), 0));
		m_labelList << text;
	        m_labelActors.push_back(vtkSmartPointer<vtkActor>::New());
		//TODO CHANGE ACTOR.
      		auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
      		m_labelActors.back()->SetMapper(mapper);

	}
    }
}
