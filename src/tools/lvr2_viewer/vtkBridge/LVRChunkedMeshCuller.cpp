
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
    //    std::cout << "Culling" << std::endl;
    double planes[24];
    vtkSmartPointer<vtkCamera> cam = ren->GetActiveCamera();
    cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes);
     double clip[2];
     cam->GetClippingRange(clip);

     std::cout << "Clipping range " << clip[0] << " " << clip[1] << std::endl;
     std::cout << "PLAAAAAAAAANES " << std::endl;
     for(int i = 0; i < 6; ++i)
     {
         int index = 4 * i;
         std::cout << planes[index] << " " <<
             planes[index + 1] << " " <<
             planes[index + 2] << " " <<
             planes[index + 3] << std::endl;

     }

    double position[3];
    cam->GetPosition(position);
    double dir[3];
    cam->GetDirectionOfProjection(dir);
    double up[3];
    cam->GetViewUp(up);

    std::vector<lvr2::BaseVector<float> > centroids;
    std::vector<size_t> indices;
    m_bridge->getActors(planes, centroids, indices);


    // new near plane
//    for(int i = 0; i < 3; ++i)
//    {
//        if(planes[4 * 4 + i])
//        {
//            base[i] = planes[4 * 4 + 3] / planes[4 * 4 + i];
//        }
//    }

    // PSEUDO
    //
    // projDir * 4 
//    base.normalize();
//    base*= 4;
//    double d = 0;
//    for(int i = 0; i < 3; ++i)
//    {
//        d = base[i] * planes[4 * 4 + i];
//    }
//    planes[4 * 4 + 3] = d;
    

    // New far plane 
//    base = BaseVector<float>(dir[0], dir[1], dir[2]);
//
//    std::cout << "View dir" <<  base << std::endl;
//    for(int i = 0; i < 3; ++i)
//    {
//        if(planes[4 * 4 + i])
//        {
//            base[i] = planes[4 * 4 + 3] / planes[4 * 4 + i];
//        }
//    }
//    base.normalize();
//    base*=20;
//    d = 0;
//    for(int i = 0; i < 3; ++i)
//    {
//        d = base[i] * planes[4 * 4 + i];
//    }
//    planes[4 * 5 + 3] = d;
//
//
    double scale = cam->GetParallelScale();

    std::cout << "VIEW ANGLE " << cam->GetViewAngle() << std::endl;
    
    cam->SetParallelScale(1.0);
    cam->SetClippingRange(0.01, 110.0);
    double planes_high[24];
    cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes_high);
    BaseVector<float> base(dir[0], dir[1], dir[2]);
    std::cout << "Dir " << base << std::endl;
    std::cout << "PLAAAAAAANES 2 " << std::endl;
    for(int i = 0; i < 6; ++i)
    {
        int index = 4 * i;
        std::cout << planes_high[index] << " " <<
                     planes_high[index + 1] << " " <<
                     planes_high[index + 2] << " " <<
                     planes_high[index + 3] << std::endl;
    }

    // Create a sphere actor to represent the current focal point
    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();    
    BaseVector<float> pos_vec(position[0], position[1], position[2]);
    pos_vec = pos_vec +  (base *110.0f) ;
    sphereSource->SetCenter(pos_vec[0] , pos_vec[1],pos_vec[2] );
    
    sphereSource->SetRadius(1.0);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(sphereSource->GetOutputPort());

    vtkSmartPointer<vtkActor> sphereActor = vtkSmartPointer<vtkActor>::New();
    sphereActor->SetMapper(mapper);
    ren->AddActor(sphereActor);

    vtkSmartPointer<vtkSphereSource> sphereSource2 = vtkSmartPointer<vtkSphereSource>::New();    
    BaseVector<float> pos_vec2(position[0], position[1], position[2]);
    pos_vec2 = pos_vec +  (base *1.0f) ;
    sphereSource->SetCenter(pos_vec[0] , pos_vec[1],pos_vec[2] );
    
    sphereSource->SetRadius(4.0);

    vtkSmartPointer<vtkPolyDataMapper> mapper2 = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(sphereSource->GetOutputPort());

    vtkSmartPointer<vtkActor> sphereActor2 = vtkSmartPointer<vtkActor>::New();
    sphereActor2->SetMapper(mapper2);
    ren->AddActor(sphereActor2);


    cam->SetParallelScale(scale);
    cam->SetClippingRange(clip[0], clip[1]);
     //cam->GetClippingRange(clip);



    std::vector<lvr2::BaseVector<float> > centroids2;
    std::vector<size_t> indices2;
    m_bridge->getActors(planes_high, centroids2, indices2);

    BoundingBox<BaseVector<float> > highResArea;
    std::cout << "Indices size() " << indices.size() << " " << indices2.size() << std::endl;

    for(size_t i = 0; i < centroids2.size(); ++i)
    {
            highResArea.expand(centroids2[i]);   
    }

    //for(size_t i = 0; i < centroids.size(); ++i)
    //{
    //    double distance = std::pow((position[0] - centroids[i][0]), 2) +
    //                      std::pow((position[1] - centroids[i][1]), 2) +
    //                      std::pow((position[2] - centroids[i][2]), 2);

    //    distance = std::sqrt(distance);
    //    //distance -= std::abs(clip[0]) - 10.0;
    //    std::cout << "Distance " << distance << std::endl;
    //    if(distance < 120)
    //    {
    //        highResArea.expand(centroids[i]);   
    //    }
    //}

//    std::cout << "HIGHRES area " << highResArea << std::endl;
    //m_bridge->fetchHighRes(position, dir, up);
    m_bridge->fetchHighRes(highResArea);

    vtkActorCollection* actors = ren->GetActors();
    actors->InitTraversal();

    for(vtkIdType i = 0; i < actors->GetNumberOfItems(); i++)
    {
        vtkActor* nextActor = actors->GetNextActor();
//        if(nextActor->IsA("MeshChunkActor"))
//        { 
              nextActor->VisibilityOff();
          
//        }
    }
    sphereActor->VisibilityOn();
    sphereActor2->VisibilityOn();
    //LVRBoundingBoxBridge bbridge(highResArea);
    //ren->AddActor(bbridge.getActor());
    //bbridge.setVisibility(true);

   

    std::cout << "low res size " << lowRes.size() << " highRes size " << highRes.size() << std::endl;
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

//    for(auto& it : highRes)
//    {
//        it.second->VisibilityOn();
//        lowRes.at(it.first)->VisibilityOff();
//    }

    //double distance_threshold;


    //vtkActorCollection* actors = ren->GetActors();
    //actors->InitTraversal();

    //int j = 0;
    //int k = 0;
    //for(vtkIdType i = 0; i < actors->GetNumberOfItems(); i++)
    //{
    //    vtkActor* nextActor = actors->GetNextActor();
    //    if(nextActor->IsA("MeshChunkActor"))
    //    { 
    //        k++;
    //        if(std::find(indices.begin(),
    //                    indices.end(),
    //                    static_cast<MeshChunkActor*>(nextActor)->getID()) == indices.end())                                                                           
    //        {
    //            nextActor->VisibilityOff();
    //        }
    //        else
    //        {
    //            if(static_cast<MeshChunkActor*>(nextActor)->getRender())
    //            {
    //            nextActor->VisibilityOn();
    //            j++;
    //            }
    //        }
    //    }
    //}
    }
//    std::cout << "j: " << j << " k: " << k << std::endl;
}

}
