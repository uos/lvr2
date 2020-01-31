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
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>

#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

LVRCamDataItem::LVRCamDataItem(
    ScanImage& data,
    std::shared_ptr<ScanDataManager> sdm,
    size_t cam_id,
    vtkSmartPointer<vtkRenderer> renderer,
    QString name,
    QTreeWidgetItem *parent
)
: QTreeWidgetItem(parent, LVRCamDataItemType)
{
    m_pItem  = nullptr;
    m_cvItem = nullptr;
    m_data   = data;
    m_name   = name;
    m_sdm    = sdm;
    m_cam_id  = cam_id;
    m_renderer = renderer;
 
    // m_matrix = m_data.m_extrinsics;

    // change this to not inverse
    bool dummy;
    m_matrix = m_data.extrinsics;//.inv(dummy);

    // set Transform from 
    setTransform(m_matrix);

    // TODO: How do we get the intrincs into the item
    // using the new structure??
    m_intrinsics = Intrinsicsd::Identity();
    

    // init pose
    double pose[6];
    extrinsicsToEuler<double>(m_data.extrinsics, pose);

    m_pose.x = pose[0];
    m_pose.y = pose[1];
    m_pose.z = pose[2];
    m_pose.r = pose[3]  * 57.295779513;
    m_pose.t = pose[4]  * 57.295779513;
    m_pose.p = pose[5]  * 57.295779513;

    m_pItem = new LVRPoseItem(ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model))), this);

    m_cvItem = new LVRCvImageItem(sdm, renderer, "Image", this);

    m_pItem->setPose(m_pose);

    m_frustrum_actor = genFrustrum(0.1);
    
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
        std::cout << "ADD ACTOR" << std::endl;
        m_renderer->AddActor(m_frustrum_actor);
    } else {
        m_renderer->RemoveActor(m_frustrum_actor);
    }
}

Transformd LVRCamDataItem::getGlobalTransform()
{
    Transformd ret = m_matrix;
    QTreeWidgetItem* parent_it = parent();

    while(parent_it != NULL)
    {

        auto transform_obj = dynamic_cast< Transformable* >(parent_it);
        if(transform_obj)
        {
            ret = ret * transform_obj->getTransform();
        }
        parent_it = parent_it->parent();
    }

    return ret;
}

void LVRCamDataItem::setCameraView()
{
    auto cam = m_renderer->GetActiveCamera();

    double x,y,z;
    cam->GetPosition(x,y,z);

    Transformd T = getGlobalTransform();

    // T.transpose();


    Vector3d cam_origin(0.0, 0.0, -1.0);
    Vector3d view_up(1.0, 0.0, 0.0);
    Vector3d focal_point(0.0, 0.0, 0.0);


    cam_origin = lvr2::multiply(T, cam_origin);
    view_up = lvr2::multiply(T, view_up);
    focal_point = lvr2::multiply(T, focal_point);

    cam->SetPosition(cam_origin.x(), cam_origin.y(), cam_origin.z());
    cam->SetFocalPoint(focal_point.x(), focal_point.y(), focal_point.z());
    cam->SetViewUp(view_up.x(), view_up.y(), view_up.z());

    //  TODO: set intrinsics

}

std::vector<Vector3d > LVRCamDataItem::genFrustrumLVR(float scale)
{
    Transformd T = getGlobalTransform().transpose();
    
    

    std::vector<Vector3d > cv_pixels;

    // TODO change this. get size of image

    Intrinsicsd intrisics_corrected = Intrinsicsd::Identity();

    intrisics_corrected(0,0) = m_intrinsics(1,1);
    intrisics_corrected(1,1) = m_intrinsics(0,0);
    intrisics_corrected(0,2) = m_intrinsics(1,2);
    intrisics_corrected(1,2) = m_intrinsics(0,2);

    int u_max = intrisics_corrected(0, 2) * 2;
    int v_max = intrisics_corrected(1, 2) * 2;

    Intrinsicsd cam_mat_inv = intrisics_corrected.inverse();
    

    std::cout << "u,v max: "<< u_max << "," << v_max << std::endl;
    // std::cout << v_max << std::endl;

    // opencv x,y,z
    // top left
    cv_pixels.push_back({0.0, 0.0, 1.0});
    // bottom left
    cv_pixels.push_back({0.0, double(v_max), 1.0});
    // bottem right
    cv_pixels.push_back({double(u_max), double(v_max), 1.0});
    // top right
    cv_pixels.push_back({double(u_max), 0.0, 1.0});


    // generate frustrum points
    std::vector<Vector3d > lvr_points;


    // origin
    lvr_points.push_back({0.0, 0.0, 0.0});

    std::cout << "cam space: " << std::endl;
    for(int i=0; i<cv_pixels.size(); i++)
    {
        Vector3d pixel = cv_pixels[i];
        //Vector3d p = lvr2::multiply(cam_mat_inv, pixel);

        Vector3d p = cam_mat_inv * pixel;

        // Vector3d p = pixel;


        Vector3d tmp = openCvToLvr(p);

        std::cout << tmp.transpose() << std::endl;

        tmp *= scale;
        lvr_points.push_back(tmp);
    }

    std::cout << "world space:" << std::endl;
    // transform frustrum
    for(int i=0; i<lvr_points.size(); i++)
    {
        // lvr_points[i] = lvr2::multiply(T, lvr_points[i]);
        std::cout << lvr_points[i].transpose() << std::endl;
        // lvr_points[i] = lvr_points[i];
    }

    return lvr_points;
}

vtkSmartPointer<vtkActor> LVRCamDataItem::genFrustrum(float scale)
{

    std::vector<Vector3d > lvr_points = genFrustrumLVR(scale);

    // Setup points
    vtkSmartPointer<vtkPoints> points =
        vtkSmartPointer<vtkPoints>::New();

    // convert to vtk
    points->SetNumberOfPoints(lvr_points.size());

    for(int i=0; i<lvr_points.size(); i++)
    {
        auto p = lvr_points[i];
        points->SetPoint(i, p.x(), p.y(), p.z());
    }

    // // Define some colors
    unsigned char white[3] = {255, 255, 255}; // origin
    unsigned char red[3] = {255, 0, 0}; // top left
    unsigned char green[3] = {0, 255, 0}; // bottom left
    unsigned char blue[3] = {0, 0, 255}; // bottom right
    unsigned char yellow[3] = {255, 255, 0}; // top right

    // // Setup the colors array
    vtkSmartPointer<vtkUnsignedCharArray> colors =
        vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetNumberOfTuples(lvr_points.size());
    colors->SetName("Colors");

#if VTK_MAJOR_VERSION < 7
    colors->SetTupleValue(0, white);
    colors->SetTupleValue(1, red);
    colors->SetTupleValue(2, green);
    colors->SetTupleValue(3, blue);
    colors->SetTupleValue(4, yellow);
#else
    colors->SetTypedTuple(0, white); // no idea how the new method is called
    colors->SetTypedTuple(1, red);
    colors->SetTypedTuple(2, green);
    colors->SetTypedTuple(3, blue);
    colors->SetTypedTuple(4, yellow);
#endif

    // Create a triangle
    vtkSmartPointer<vtkCellArray> triangles =
        vtkSmartPointer<vtkCellArray>::New();

    
    vtkSmartPointer<vtkTriangle> triangle =
        vtkSmartPointer<vtkTriangle>::New();

    // left plane

    triangle->GetPointIds()->SetId(0, 0);
    triangle->GetPointIds()->SetId(1, 1);
    triangle->GetPointIds()->SetId(2, 2);

    triangles->InsertNextCell(triangle);

    // bottom plane
    triangle->GetPointIds()->SetId(0, 0);
    triangle->GetPointIds()->SetId(1, 2);
    triangle->GetPointIds()->SetId(2, 3);

    triangles->InsertNextCell(triangle);

    // right plane
    triangle->GetPointIds()->SetId(0, 0);
    triangle->GetPointIds()->SetId(1, 3);
    triangle->GetPointIds()->SetId(2, 4);

    triangles->InsertNextCell(triangle);

    // top plane
    triangle->GetPointIds()->SetId(0, 0);
    triangle->GetPointIds()->SetId(1, 4);
    triangle->GetPointIds()->SetId(2, 1);

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

LVRCamDataItem::~LVRCamDataItem()
{
    // we don't want to do delete m_bbItem, m_pItem and m_pcItem here
    // because QTreeWidgetItem deletes its childs automatically in its destructor.
}

} // namespace lvr2
