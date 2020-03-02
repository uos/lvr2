
#include "LVRChunkedMeshCuller.hpp"

//#include "MeshChunkActor.hpp"

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
        std::unordered_map<size_t, vtkSmartPointer<vtkActor>> lowRes = m_bridge->getLowResActors();
        std::unordered_map<size_t, vtkSmartPointer<vtkActor>> highRes = m_bridge->getHighResActors();

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

        cam->SetParallelScale(1.0);
        cam->SetClippingRange(0.01, m_highResDistance);
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
        //std::cout << highResArea << std::endl;
        //std::cout << highResArea.getVolume() << std::endl;

        //BaseVector<float> eye(position[0], position[1], position[2]);
        //BaseVector<float> direction(dir[0], dir[1], dir[2]);
        //direction.normalize();
        //BaseVector<float> upVector(up[0], up[1], up[2]);
        //upVector.normalize();
        //BaseVector<float> perp = upVector.cross(direction);
        //perp.normalize();

        //highResArea.expand(eye + (direction * 1.2 * 150.0));
        //highResArea.expand(eye + (direction * ((-1) * 100.0/2.0)));
        //highResArea.expand(eye + (upVector * 100.0));
        //highResArea.expand(eye + (upVector * ((-1) * 100.0)));
        //highResArea.expand(eye + (perp * 100.0));
        //highResArea.expand(eye + (perp * ((-1) * 100.0)));

        m_bridge->fetchHighRes(highResArea, indices2, centroids2);

        //vtkActorCollection* actors = ren->GetActors();
        //actors->InitTraversal();


        if(lowRes.size() > 0)
        {
            for(size_t i = 0; i < indices.size(); ++i)
            {
                lowRes.at(indices[i])->VisibilityOn();
            }
        }

        for(auto& it: highRes)
        {
            if(std::find(indices2.begin(), indices2.end(), it.first) != indices2.end())
            {
                if(it.second)
                {
                    it.second->VisibilityOn();
                }
                if(lowRes.size() > 0)
                {
                    lowRes.at(it.first)->VisibilityOff();
                }
            }
            else
            {
                if(it.second)
                {
                    it.second->VisibilityOff();
                }
            }
        }
        return 0.0;
    }
}
