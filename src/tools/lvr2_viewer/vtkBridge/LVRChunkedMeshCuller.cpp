
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
    //    std::cout << "Culling" << std::endl;
    double planes[24];
    vtkSmartPointer<vtkCamera> cam = ren->GetActiveCamera();
    cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes);
    double x, y, z;
    cam->GetPosition(x, y, z);

    std::vector<lvr2::BaseVector<float> > centroids;
    std::vector<size_t> indices;
    m_bridge->getActors(planes, centroids, indices);

    

    double distance_threshold;


    vtkActorCollection* actors = ren->GetActors();
    actors->InitTraversal();

//    std::cout << indices.size() << " " << actors->GetNumberOfItems() << std::endl;
    int j = 0;
    int k = 0;
    for(vtkIdType i = 0; i < actors->GetNumberOfItems(); i++)
    {
        vtkActor* nextActor = actors->GetNextActor();
        if(nextActor->IsA("MeshChunkActor"))
        { 
            k++;
            if(std::find(indices.begin(),
                        indices.end(),
                        static_cast<MeshChunkActor*>(nextActor)->getID()) == indices.end())                                                                           
            {
                nextActor->VisibilityOff();
            }
            else
            {
                nextActor->VisibilityOn();
                j++;
            }
        }
    }

//    std::cout << "j: " << j << " k: " << k << std::endl;
}

}
