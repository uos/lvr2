#include "LVRCamDataItem.hpp"
#include "LVRModelItem.hpp"
#include "LVRItemTypes.hpp"
#include "LVRScanDataItem.hpp"

#include <vtkVersion.h>
#include <vtkFrustumSource.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkPlanes.h>
#include <vtkMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkTriangle.h>
#include <vtkCellArray.h>

namespace lvr2
{

LVRCamDataItem::LVRCamDataItem(CamData data, std::shared_ptr<ScanDataManager> sdm, size_t idx, vtkSmartPointer<vtkRenderer> renderer, QString name, QTreeWidgetItem *parent) : QTreeWidgetItem(parent, LVRCamDataItemType)
{
    // m_pItem  = nullptr;
    m_data   = data;
    m_name   = name;
    m_sdm    = sdm;
    m_idx    = idx;
    m_renderer = renderer;

    // init pose
    float pose[6];
    m_data.m_extrinsics.transpose();
    m_data.m_extrinsics.toPostionAngle(pose);
    m_data.m_extrinsics.transpose();
    m_matrix = m_data.m_extrinsics;

    m_pose.x = pose[0];
    m_pose.y = pose[1];
    m_pose.z = pose[2];
    m_pose.r = pose[3]  * 57.295779513;
    m_pose.t = pose[4]  * 57.295779513;
    m_pose.p = pose[5]  * 57.295779513;

    m_pItem = new LVRPoseItem(ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model))), this);

    m_pItem->setPose(m_pose);

    Matrix4<BaseVector<float> > global_transform = getTransformation();

    // search for upper scan items to get global transformation

    // init bb
    // m_bb = BoundingBoxBridgePtr(new LVRBoundingBoxBridge(m_data.m_boundingBox));
    // m_bbItem = new LVRBoundingBoxItem(m_bb, "Bounding Box", this);
    // renderer->AddActor(m_bb->getActor());
    // m_bb->setPose(m_pose);

    m_frustrum_actor = genFrustrum();
    // renderer->AddActor(m_frustrum_actor);
    // load data
    reload(renderer);

    setText(0, m_name);
    setCheckState(0, Qt::Unchecked );
}

void LVRCamDataItem::reload(vtkSmartPointer<vtkRenderer> renderer)
{
    
}

void LVRCamDataItem::setVisibility(bool visible)
{
    if(checkState(0) && visible)
    {
        m_renderer->AddActor(m_frustrum_actor);
    } else {
        m_renderer->RemoveActor(m_frustrum_actor);
    }
}

Matrix4<BaseVector<float> > LVRCamDataItem::getTransformation()
{
    Matrix4<BaseVector<float> > ret;
    QTreeWidgetItem* parent_it = parent();

    while(parent_it != NULL && parent_it->type() != LVRScanDataItemType)
    {
        parent_it = parent_it->parent();
    }
    if(parent_it)
    {
        LVRScanDataItem* item = (LVRScanDataItem*)parent_it;
        // TODO: check this
        Matrix4<BaseVector<float> > global_transform = item->getTransformation();
        return global_transform * m_matrix;
    }
    return m_matrix;
}

vtkSmartPointer<vtkActor> LVRCamDataItem::genFrustrum()
{
    // TODO better frustrum

    Matrix4<BaseVector<float> > T;
    // Setup points
    vtkSmartPointer<vtkPoints> points =
        vtkSmartPointer<vtkPoints>::New();

    // generate frustrum points
    std::vector<Vector<BaseVector<float> > > lvr_points;
    
    lvr_points.push_back({10.0, 0.0, 0.0});
    lvr_points.push_back({0.0, 0.0, 0.0});
    lvr_points.push_back({0.0, 10.0, 0.0});

    // transform frustrum
    for(int i=0; i<lvr_points.size(); i++)
    { 
        lvr_points[i] = T * lvr_points[i];
    }

    // convert to vtk
    points->SetNumberOfPoints(lvr_points.size());

    for(int i=0; i<lvr_points.size(); i++)
    {
        auto p = lvr_points[i];
        points->SetPoint(i, p.x, p.y, p.z);
    }

    // // Define some colors
    unsigned char red[3] = {255, 0, 0};
    unsigned char green[3] = {0, 255, 0};
    unsigned char blue[3] = {0, 0, 255};

    // // Setup the colors array
    vtkSmartPointer<vtkUnsignedCharArray> colors =
        vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetNumberOfTuples(lvr_points.size());
    colors->SetName("Colors");

#if VTK_MAJOR_VERSION < 7
    colors->SetTupleValue(0, red);
    colors->SetTupleValue(1, green);
    colors->SetTupleValue(2, blue);
#else
    colors->SetTypedTuple(0, red); // no idea how the new method is called
    colors->SetTypedTuple(1, green);
    colors->SetTypedTuple(2, blue);
#endif

    // Create a triangle
    vtkSmartPointer<vtkCellArray> triangles =
        vtkSmartPointer<vtkCellArray>::New();

    
    vtkSmartPointer<vtkTriangle> triangle =
        vtkSmartPointer<vtkTriangle>::New();

    
    triangle->GetPointIds()->SetId(0, 0);
    triangle->GetPointIds()->SetId(1, 1);
    triangle->GetPointIds()->SetId(2, 2);

    triangles->InsertNextCell(triangle);

    // Create a polydata object and add everything to it
    vtkSmartPointer<vtkPolyData> polydata =
        vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetPolys(triangles);
    polydata->GetPointData()->SetScalars(colors);

    // // Visualize
    vtkSmartPointer<vtkPolyDataMapper> mapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInputConnection(polydata->GetProducerPort());
#else
    mapper->SetInputData(polydata);
#endif
    vtkSmartPointer<vtkActor> actor =
        vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    return actor;
}

// void LVRCamDataItem::dataChanged(){
//     std::cout << "test" << std::endl;
// }

LVRCamDataItem::~LVRCamDataItem()
{
    // we don't want to do delete m_bbItem, m_pItem and m_pcItem here
    // because QTreeWidgetItem deletes its childs automatically in its destructor.
}

} // namespace lvr2
