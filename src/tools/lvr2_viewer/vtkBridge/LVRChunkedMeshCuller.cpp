#include "LVRChunkedMeshCuller.hpp"
#include <omp.h>

using namespace lvr2;

ChunkedMeshCuller::ChunkedMeshCuller(std::string file):
    m_chunkManager(file)
{
}

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

    double x_min, y_min, z_min = 10e6;
    double x_max, y_max, z_max = -10e6;

    #omp pragma parallel for collapse(3) reduction(max:x_max, y_max, z_max) reduction(max:x_max, y_max, z_max)
    for(size_t i = 0; i < 1; ++i)
    {
        // left/right
        double *U = planes + (i * 4);
        for(size_t j = 2; j < 3; ++j)
        {
            // bottom/top
            double *V = planes + (j * 4);
            for(size_t k = 4; k < 5; ++k)
            {
                double *W = planes + (k * 4);
                double det = sarrus(U, V, W, A, B, C);
                double x   = sarrus(U, V, W, D, B, C) / ((-1) * det);
                double y   = sarrus(U, V, W, A, D, C) / ((-1) * det);
                double z   = sarrus(U, V, W, A, B, D) / ((-1) * det);
                x_min = std::min(x_min, x);
                y_min = std::min(y_min, y);
                z_min = std::min(z_min, z);
                x_max = std::max(x_max, x);
                y_max = std::max(y_max, y);
                z_max = std::max(z_max, z);
            }
        }
    }
}

double ChunkedMeshCuller::Cull(vtkRenderer *ren, vtkProp **propList, int &listLength, int &initialized)
{
    double planes[24];
    ren->GetActiveCamera()->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes);

    lvr2::BoundingBox<BaseVector<float> > aabb = frustumToBB(planes);
    m_chunkManager.extractArea(aabb, m_chunks);

    vtkActorCollection* actors = ren->GetActors();
    


    // if not
}
