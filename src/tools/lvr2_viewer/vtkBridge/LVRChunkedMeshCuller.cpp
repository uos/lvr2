
#include "LVRChunkedMeshCuller.hpp"

#include "MeshChunkActor.hpp"

#include "LVRBoundingBoxBridge.hpp"

#include <omp.h>

using namespace lvr2;

double ChunkedMeshCuller::sarrus(const double U[4],
                                 const double V[4],
                                 const double W[4],
                                 size_t A,
                                 size_t B,
                                 size_t C)
{
   double det = U[A] * V[B] * W[C] +
                U[B] * V[C] * W[A] +
                U[C] * V[A] * W[B] -
                W[A] * V[B] * U[C] -
                W[B] * V[C] * U[A] -
                W[C] * V[A] * U[B];

    return det;
}

BoundingBox<BaseVector<float> > ChunkedMeshCuller::frustumToBB(double planes[24])
{
    // planes: left right bottom top far near
    // intersections:
    //  left top far
    //  left top near
    //  left bottom far
    //  left bottom near
    //  right top far
    //  right top near
    //  right bottom far
    //  right bottom near
    const size_t A = 0;
    const size_t B = 1;
    const size_t C = 2;
    const size_t D = 3;

    double x_min, y_min, z_min;
    x_min = y_min = z_min = 20e6;
    double x_max, y_max, z_max;
    x_max = y_max = z_max = (-1) * 20e6;
    #pragma omp parallel for collapse(3) reduction(min:x_min, y_min, z_min) reduction(max:x_max, y_max, z_max)
    for(size_t i = 0; i < 2; ++i)
    {
        // left/right
        for(size_t j = 2; j < 4; ++j)
        {
            // bottom/top
                for(size_t k = 4; k < 6; ++k)
            {
                double *U = planes + (i * 4);
                double *W = planes + (k * 4);
                double *V = planes + (j * 4);
                double det = sarrus(U, V, W, A, B, C);

                // COORDINATE SYSTEM DAFUQ?!
                double x   = sarrus(U, V, W, D, B, C) / ((-1) * det);
                double y   = sarrus(U, V, W, A, D, C) / ((-1) * det);
                double z   = sarrus(U, V, W, A, B, D) / ((-1) * det);
                //std::cout << x << y << z << std::endl;
                x_min = std::min(x_min, x);
                y_min = std::min(y_min, y);
                z_min = std::min(z_min, z);
                x_max = std::max(x_max, x);
                y_max = std::max(y_max, y);
                z_max = std::max(z_max, z);
            }
        }
    }
    BaseVector<float> minbb(x_min/20.0, y_min/20.0, z_min/20.0);
    BaseVector<float> maxbb(x_max/20.0, y_max/20.0, z_max/20.0);
    return BoundingBox<BaseVector<float> >(minbb, maxbb);
//     for(size_t i = 0; i < 6; ++i)
//     {
//        x_min = std::min(x_min, planes[i * 4 + 0]);
//        y_min = std::min(y_min, planes[i * 4 + 1]);
//        z_min = std::min(z_min, planes[i * 4 + 2]);
//        x_max = std::max(x_max, planes[i * 4 + 0]);
//        y_max = std::max(y_max, planes[i * 4 + 1]);
//        z_max = std::max(z_max, planes[i * 4 + 2]);
//     }
}

double ChunkedMeshCuller::Cull(vtkRenderer *ren, vtkProp **propList, int &listLength, int &initialized)
{
//    std::cout << "Culling" << std::endl;
    double planes[24];
    ren->GetActiveCamera()->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes);

//    ren->RemoveAllViewProps();
//    std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > chunks;
    std::vector<size_t> indices;
    m_bridge->getActors(planes, indices);

    std::cout << "got " << indices.size() << " indices" << std::endl;
    
//    for(auto& actor: chunks)
//    {
//        ren->AddActor(actor.second);
//    }
//    for(auto& actor: chunks)
//    {
//        actor->VisibilityOn();
////        ren->AddActor(actor);
//    }


    vtkActorCollection* actors = ren->GetActors();
    actors->InitTraversal();

//    std::cout << "start iterating " << std::endl;
//   ren->SetAllocatedRenderTime(3.0);
    int j = 0;
//    #pragma omp parallel for shared(j) reduction(+:j) reduction(>:j)
    for(vtkIdType i = 0; i < actors->GetNumberOfItems(); i++)
    {
        vtkActor* nextActor = actors->GetNextActor();
        if(nextActor->IsA("MeshChunkActor"))
        {
            if(std::find(indices.begin(), indices.end(), static_cast<MeshChunkActor*>(nextActor)->getID()) == indices.end())
            {
//                std::cout << "Set visibility off" << std::endl;
                nextActor->VisibilityOff();
//                chunks.erase(static_cast<MeshChunkActor*>(nextActor)->getID());
            }
            else
            {
                if(j > 3000)
                {
                    continue;
                }

                //std::cout << "set Visibilityoff" << std::endl;
                nextActor->VisibilityOn();
                j++;
            }
        }
        else
        {
            //std::cout << "No MeshChunkActor" << std::endl;
            // what to do with bounding boxes... 
        }
    }

    // if not
}
