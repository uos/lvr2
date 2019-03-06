#include "LVRBoundingBoxBridge.hpp"
#include "LVRModelBridge.hpp"

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPoints.h>
#include <vtkProperty.h>
#include <vtkLine.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkTransform.h>

namespace lvr2
{

LVRBoundingBoxBridge::LVRBoundingBoxBridge(BoundingBox<Vec> bb) : m_boundingBox(bb)
{
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints>   points = vtkSmartPointer<vtkPoints>::New();

    Vec min = m_boundingBox.getMin();
    Vec max = m_boundingBox.getMax();

    float pt[3];
    // front
    // 0: bottom left
     pt[0] = min.x, pt[1] = min.y, pt[2] = min.z;
    points->InsertNextPoint(pt);
    // 1: bottom right
    pt[0] = max.x, pt[1] = min.y, pt[2] = min.z;
    points->InsertNextPoint(pt);
    // 2: top left
    pt[0] = min.x, pt[1] = max.y, pt[2] = min.z;
    points->InsertNextPoint(pt);
    // 3: top right
    pt[0] = max.x, pt[1] = max.y, pt[2] = min.z;
    points->InsertNextPoint(pt);

    // back
    // 4: bottom left
    pt[0] = min.x, pt[1] = min.y, pt[2] = max.z;
    points->InsertNextPoint(pt);
    // 5: bottom right
    pt[0] = max.x, pt[1] = min.y, pt[2] = max.z;
    points->InsertNextPoint(pt);
    // 6: top left
    pt[0] = min.x, pt[1] = max.y, pt[2] = max.z;
    points->InsertNextPoint(pt);
    // 7: top right
    pt[0] = max.x, pt[1] = max.y, pt[2] = max.z;
    points->InsertNextPoint(pt);

    polyData->SetPoints(points);

    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();

    // Front
    // 0
    vtkSmartPointer<vtkLine> line0 = vtkSmartPointer<vtkLine>::New();
    line0->GetPointIds()->SetId(0, 0);
    line0->GetPointIds()->SetId(1, 1);
    lines->InsertNextCell(line0);
    // 1
    vtkSmartPointer<vtkLine> line1 = vtkSmartPointer<vtkLine>::New();
    line1->GetPointIds()->SetId(0, 0);
    line1->GetPointIds()->SetId(1, 2);
    lines->InsertNextCell(line1);
    // 2
    vtkSmartPointer<vtkLine> line2 = vtkSmartPointer<vtkLine>::New();
    line2->GetPointIds()->SetId(0, 3);
    line2->GetPointIds()->SetId(1, 1);
    lines->InsertNextCell(line2);
    // 3
    vtkSmartPointer<vtkLine> line3 = vtkSmartPointer<vtkLine>::New();
    line3->GetPointIds()->SetId(0, 3);
    line3->GetPointIds()->SetId(1, 2);
    lines->InsertNextCell(line3);

    // Back
    // 4
    vtkSmartPointer<vtkLine> line4 = vtkSmartPointer<vtkLine>::New();
    line4->GetPointIds()->SetId(0, 4);
    line4->GetPointIds()->SetId(1, 5);
    lines->InsertNextCell(line4);
    // 5
    vtkSmartPointer<vtkLine> line5 = vtkSmartPointer<vtkLine>::New();
    line5->GetPointIds()->SetId(0, 4);
    line5->GetPointIds()->SetId(1, 6);
    lines->InsertNextCell(line5);
    // 6
    vtkSmartPointer<vtkLine> line6 = vtkSmartPointer<vtkLine>::New();
    line6->GetPointIds()->SetId(0, 7);
    line6->GetPointIds()->SetId(1, 5);
    lines->InsertNextCell(line6);
    // 7
    vtkSmartPointer<vtkLine> line7 = vtkSmartPointer<vtkLine>::New();
    line7->GetPointIds()->SetId(0, 7);
    line7->GetPointIds()->SetId(1, 6);
    lines->InsertNextCell(line7);

    // connection front back
    // 8
    vtkSmartPointer<vtkLine> line8 = vtkSmartPointer<vtkLine>::New();
    line8->GetPointIds()->SetId(0, 0);
    line8->GetPointIds()->SetId(1, 4);
    lines->InsertNextCell(line8);
    // 9
    vtkSmartPointer<vtkLine> line9 = vtkSmartPointer<vtkLine>::New();
    line9->GetPointIds()->SetId(0, 1);
    line9->GetPointIds()->SetId(1, 5);
    lines->InsertNextCell(line9);
    // 10
    vtkSmartPointer<vtkLine> line10 = vtkSmartPointer<vtkLine>::New();
    line10->GetPointIds()->SetId(0, 2);
    line10->GetPointIds()->SetId(1, 6);
    lines->InsertNextCell(line10);
    // 11
    vtkSmartPointer<vtkLine> line11 = vtkSmartPointer<vtkLine>::New();
    line11->GetPointIds()->SetId(0, 3);
    line11->GetPointIds()->SetId(1, 7);
    lines->InsertNextCell(line11);

    polyData->SetLines(lines);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    m_actor = vtkSmartPointer<vtkActor>::New();
    m_actor->SetMapper(mapper);

    setColor(0.0, 1.0, 0.0);
}

void LVRBoundingBoxBridge::setColor(double r, double g, double b)
{
    double color[] = {r, g, b};
    m_actor->GetProperty()->SetColor(color);
}

void LVRBoundingBoxBridge::setPose(const Pose &pose)
{
    vtkSmartPointer<vtkTransform> transform =  vtkSmartPointer<vtkTransform>::New();
    transform->PostMultiply();
    transform->RotateX(pose.r);
    transform->RotateY(pose.t);
    transform->RotateZ(pose.p);
    transform->Translate(pose.x, pose.y, pose.z);

    m_actor->SetUserTransform(transform);
}

void LVRBoundingBoxBridge::setVisibility(bool visible)
{
    if (visible)
    {
        m_actor->VisibilityOn();
    }
    else
    {
        m_actor->VisibilityOff();
    }
}

} // namespace lvr2
