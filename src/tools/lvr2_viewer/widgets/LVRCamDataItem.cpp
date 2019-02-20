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

namespace lvr2
{

LVRCamDataItem::LVRCamDataItem(
    CamData data,
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

    m_matrix = m_data.m_extrinsics;

    // init pose
    float pose[6];
    m_data.m_extrinsics.transpose();
    m_data.m_extrinsics.toPostionAngle(pose);
    m_data.m_extrinsics.transpose();
    

    m_pose.x = pose[0];
    m_pose.y = pose[1];
    m_pose.z = pose[2];
    m_pose.r = pose[3]  * 57.295779513;
    m_pose.t = pose[4]  * 57.295779513;
    m_pose.p = pose[5]  * 57.295779513;

    m_pItem = new LVRPoseItem(ModelBridgePtr(new LVRModelBridge( ModelPtr( new Model))), this);

    m_cvItem = new LVRCvImageItem(sdm, renderer, "Image", this);

    m_pItem->setPose(m_pose);

    // Matrix4<BaseVector<float> > global_transform = getTransformation();
    // std::cout << m_matrix << std::endl;
    // std::cout << global_transform << std::endl;

    // search for upper scan items to get global transformation

    m_frustrum_actor = genFrustrum(2.0);
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
        std::cout << "got upper transformation" << std::endl;
        // TODO: check this
        Matrix4<BaseVector<float> > global_transform = item->getTransformation();
        bool dummy;
        return global_transform.inv(dummy) * m_matrix;
    }else{
        std::cout << "doesnt found upper transform" << std::endl;
    }
    return m_matrix;
}

void LVRCamDataItem::setCameraView()
{
    std::cout << "move rendering camera to this" << std::endl;
    auto cam = m_renderer->MakeCamera();

    double x,y,z;
    cam->GetPosition(x,y,z);
    std::cout << "current pos: " << x << ", " << y << ", " << z << std::endl;

    Matrix4<BaseVector<float> > transform = getTransformation();

    vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
    vtkSmartPointer<vtkMatrix4x4> m = vtkSmartPointer<vtkMatrix4x4>::New();

    // For some reason we have to copy the matrix
    // values manually...
    int j = 0;
    for(int i = 0; i < 16; i++)
    {
        if((i % 4) == 0)
        {
            j = 0;
        }
        double v = transform[i];
        m->SetElement(i / 4, j, v);
        j++;
    }

    t->PostMultiply();
    t->SetMatrix(m);

    cam->ApplyTransform(t);

    m_renderer->SetActiveCamera(cam);
}

vtkSmartPointer<vtkActor> LVRCamDataItem::genFrustrum(float scale)
{

    Matrix4<BaseVector<float> > T = getTransformation();
    bool dummy;
    Matrix4<BaseVector<float> > T_inv = T.inv(dummy);

    Matrix4<BaseVector<float> > cam_mat_inv = m_data.m_intrinsics.inv(dummy);
    cam_mat_inv.transpose();

    std::vector<Vector<BaseVector<float> > > lvr_pixels;

    // TODO change this. get size of image

    int v_max = m_data.m_intrinsics[2] * 2;
    int u_max = m_data.m_intrinsics[6] * 2;

    // top left
    lvr_pixels.push_back({0.0, 0.0, 1.0});
    // bottom left
    lvr_pixels.push_back({float(v_max), 0.0, 1.0});
    // bottem right
    lvr_pixels.push_back({float(v_max), float(u_max), 1.0});
    // top right
    lvr_pixels.push_back({0.0, float(u_max), 1.0});


    // Setup points
    vtkSmartPointer<vtkPoints> points =
        vtkSmartPointer<vtkPoints>::New();

    // generate frustrum points
    std::vector<Vector<BaseVector<float> > > lvr_points;
    
    
    // origin
    lvr_points.push_back({0.0, 0.0, 0.0});

    for(int i=0; i<lvr_pixels.size(); i++)
    {
        Vector<BaseVector<float> > pixel = lvr_pixels[i];
        Vector<BaseVector<float> > p = cam_mat_inv * pixel;

        std::cout << p << std::endl;


        // opencv to lvr
        Vector<BaseVector<float> > tmp = {
            p.y,
            p.z,
            -p.x
        };
        tmp *= scale;
        lvr_points.push_back(tmp);
    }


    T_inv.transpose();
    T.transpose();
    // transform frustrum
    for(int i=0; i<lvr_points.size(); i++)
    {
        std::cout << "Transform point" << std::endl;
        std::cout << lvr_points[i] << std::endl;

        lvr_points[i] = T_inv * lvr_points[i];

        std::cout << lvr_points[i] << std::endl;
    }

    // convert to vtk
    points->SetNumberOfPoints(lvr_points.size());

    for(int i=0; i<lvr_points.size(); i++)
    {
        auto p = lvr_points[i];
        points->SetPoint(i, p.x, p.y, p.z);
    }

    // // Define some colors
    unsigned char white[3] = {255, 255, 255};
    unsigned char red[3] = {255, 0, 0};
    unsigned char green[3] = {0, 255, 0};
    unsigned char blue[3] = {0, 0, 255};
    unsigned char yellow[3] = {255, 255, 0};

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
