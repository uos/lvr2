
#include "LVRChunkedMeshCuller.hpp"

#include "MeshChunkActor.hpp"

#include "LVRBoundingBoxBridge.hpp"

#include <omp.h>

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
    double x, y, z;
    cam->GetPosition(x, y, z);
    double dir_x, dir_y, dir_z;
    cam->GetDirectionOfProjection(dir_x, dir_y, dir_z);
    m_bridge->fetchHighRes(x, y, z, dir_x, dir_y, dir_z);
    BaseVector<float> offset(dir_x, dir_y, dir_z);
    offset *= 30;
    
    BaseVector<float> min(x < offset[0] ? x : offset[0],
                          y < offset[1] ? y : offset[1],
                          z < offset[2] ? z : offset[2]);
    BaseVector<float> max(x > offset[0] ? x : offset[0],
                          y > offset[1] ? y : offset[1],
                          z > offset[2] ? z : offset[2]);
   
//    BaseVector<float> centroid(x, y, z);

    BoundingBox<BaseVector<float> > highResArea =  BoundingBox<BaseVector<float> >(min, max);

    //LVRBoundingBoxBridge bbridge(highResArea);
    //ren->AddActor(bbridge.getActor());
    //bbridge.setVisibility(true);

    std::vector<lvr2::BaseVector<float> > centroids;
    std::vector<size_t> indices;
    m_bridge->getActors(planes, centroids, indices);


    vtkActorCollection* actors = ren->GetActors();
    actors->InitTraversal();

    for(vtkIdType i = 0; i < actors->GetNumberOfItems(); i++)
    {
        vtkActor* nextActor = actors->GetNextActor();
        if(nextActor->IsA("MeshChunkActor"))
        { 
                nextActor->VisibilityOff();
            
        }
    }
    

    std::cout << "low res size " << lowRes.size() << " highRes size " << highRes.size() << std::endl;
    for(size_t i = 0; i < indices.size(); ++i)
    {
        if(highRes.find(indices[i]) == highRes.end())
        {
            lowRes.at(indices[i])->VisibilityOn();
        }
    }

    for(auto& it : highRes)
    {
//        lowRes.at(it.first)->VisibilityOff();
    }

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
