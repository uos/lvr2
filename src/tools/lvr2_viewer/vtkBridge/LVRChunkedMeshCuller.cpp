
#include "LVRChunkedMeshCuller.hpp"

#include "MeshChunkActor.hpp"

#include "LVRBoundingBoxBridge.hpp"

#include <omp.h>

#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>

//using namespace lvr2;

namespace lvr2{

double ChunkedMeshCuller::Cull(vtkRenderer *ren,
        vtkProp **propList,
        int &listLength,
        int &initialized)
{
    actorMap lowRes = m_bridge->getLowResActors();
    actorMap highRes = m_bridge->getHighResActors();
    if(!cull_)
    {
        return 0.0;
    }
    else{
    double planes[24];
    vtkSmartPointer<vtkCamera> cam = ren->GetActiveCamera();
    cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes);
     double clip[2];
     cam->GetClippingRange(clip);

     //for(int i = 0; i < 6; ++i)
     //{
     //    int index = 4 * i;

     //}

    double position[3];
    cam->GetPosition(position);
    double dir[3];
    cam->GetDirectionOfProjection(dir);
    double up[3];
    cam->GetViewUp(up);

    std::vector<lvr2::BaseVector<float> > centroids;
    std::vector<size_t> indices;
    m_bridge->getActors(planes, centroids, indices);


    double scale = cam->GetParallelScale();

    //std::cout << "VIEW ANGLE " << cam->GetViewAngle() << std::endl;
    
    cam->SetParallelScale(1.0);
    cam->SetClippingRange(0.01, 110.0);
    double planes_high[24];
    cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes_high);
    BaseVector<float> base(dir[0], dir[1], dir[2]);
    cam->SetParallelScale(scale);
    cam->SetClippingRange(clip[0], clip[1]);
    std::vector<lvr2::BaseVector<float> > centroids2;
    std::vector<size_t> indices2;
    m_bridge->getActors(planes_high, centroids2, indices2);

    BoundingBox<BaseVector<float> > highResArea;

    for(size_t i = 0; i < centroids2.size(); ++i)
    {
            highResArea.expand(centroids2[i]);   
    }

    m_bridge->fetchHighRes(highResArea);


    // Create a sphere actor to represent the current focal point
    //vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();    
    //BaseVector<float> pos_vec(position[0], position[1], position[2]);
    //pos_vec = pos_vec +  (base *110.0f) ;
    //sphereSource->SetCenter(pos_vec[0] , pos_vec[1],pos_vec[2] );
    //
    //sphereSource->SetRadius(1.0);

    //vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    //mapper->SetInputConnection(sphereSource->GetOutputPort());

    //vtkSmartPointer<vtkActor> sphereActor = vtkSmartPointer<vtkActor>::New();
    //sphereActor->SetMapper(mapper);
    //ren->AddActor(sphereActor);

    //vtkSmartPointer<vtkSphereSource> sphereSource2 = vtkSmartPointer<vtkSphereSource>::New();    
    //BaseVector<float> pos_vec2(position[0], position[1], position[2]);
    //pos_vec2 = pos_vec +  (base *1.0f) ;
    //sphereSource->SetCenter(pos_vec[0] , pos_vec[1],pos_vec[2] );
    //
    //sphereSource->SetRadius(4.0);

    //vtkSmartPointer<vtkPolyDataMapper> mapper2 = vtkSmartPointer<vtkPolyDataMapper>::New();
    //mapper->SetInputConnection(sphereSource->GetOutputPort());

    //vtkSmartPointer<vtkActor> sphereActor2 = vtkSmartPointer<vtkActor>::New();
    //sphereActor2->SetMapper(mapper2);
    //ren->AddActor(sphereActor2);

    
    vtkActorCollection* actors = ren->GetActors();
    actors->InitTraversal();

    #pragma omp parallel for
    for(vtkIdType i = 0; i < actors->GetNumberOfItems(); i++)
    {
        vtkActor* nextActor = actors->GetNextActor();
        if(nextActor->IsA("MeshChunkActor"))
        { 
            nextActor->VisibilityOff();
        
        }
    }
    //LVRBoundingBoxBridge bbridge(highResArea);
    //ren->AddActor(bbridge.getActor());
    //bbridge.setVisibility(true);

   
    #pragma omp parallel for
    for(size_t i = 0; i < indices.size(); ++i)
    {
        if(highRes.find(indices[i]) == highRes.end())
        {
            lowRes.at(indices[i])->VisibilityOn();
        }
        else
        {
            highRes.at(indices[i])->VisibilityOn();
        }

    }

    }
}

}
