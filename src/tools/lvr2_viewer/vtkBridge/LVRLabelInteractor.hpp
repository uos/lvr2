#ifndef LVRLABELINTERACTOR_STLYE_H
#define LVRLABELINTERACTOR_STLYE_H
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkIdTypeArray.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkRendererCollection.h>
#include <vtkProperty.h>
#include <vtkPlanes.h>
#include <vtkObjectFactory.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyData.h>
#include <vtkPointSource.h>
#include "LVRInteractorStylePolygonPick.hpp"
#include <vtkInteractorStyleDrawPolygon.h>
#include <vtkAreaPicker.h>
#include <vtkExtractGeometry.h>
#include <vtkDataSetMapper.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkIdFilter.h>
#include <QObject>
#include <vector>
#include <QStringList>
#include <memory>
#include <fstream>

// Define interaction style
//class LVRLabelInteractorStyle : public QObject, public vtkInteractorStyleDrawPolygon
class LVRLabelInteractorStyle : public QObject, public LVRInteractorStylePolygonPick
{
  Q_OBJECT
  public:
    static LVRLabelInteractorStyle* New();
    vtkTypeMacro(LVRLabelInteractorStyle,LVRInteractorStylePolygonPick);

    LVRLabelInteractorStyle();

    virtual void OnKeyUp();
    virtual void OnRightButtonDown();
    virtual void OnRightButtonUp();
    virtual void OnLeftButtonUp();
    inline void SetPoints(vtkSmartPointer<vtkPolyData> points) {m_points = points;
		m_SelectedPoints = std::vector<bool>(m_points->GetNumberOfPoints(), false);
		m_pointLabels = std::vector<uint8_t>(m_points->GetNumberOfPoints(), 0);
    }

public Q_SLOTS:
    void labelSelectedPoints(QString label);
    void extractLabel();
Q_SIGNALS:
    void pointsSelected();
  private:
    void calculateSelection(bool select);
    vtkSmartPointer<vtkIdTypeArray> m_selectedIds; 
    vtkSmartPointer<vtkPolyData> m_points;
    std::vector<bool> m_SelectedPoints;
    std::vector<std::vector<bool>> foo2;
    std::vector<vtkSmartPointer<vtkActor>> m_labelActors;
    std::vector<uint8_t> m_pointLabels;
    vtkSmartPointer<vtkActor> SelectedActor;
    vtkSmartPointer<vtkDataSetMapper> SelectedMapper;
    QStringList m_labelList;
    std::vector<std::vector<uint8_t>> colors;

};
#endif // LVRLABELINTERACTOR_STLYE_H
