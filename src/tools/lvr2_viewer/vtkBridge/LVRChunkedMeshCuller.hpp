#include <vtkCuller.h>

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>

#include "lvr2/algorithm/ChunkManager.hpp"

#include <string>



namespace lvr2 {
    class ChunkedMeshCuller: public vtkCuller
    {
        public:
            ChunkedMeshCuller(std::string file);
            virtual double Cull(vtkRenderer *ren, vtkProp **propList, int &listLength, int &initialized) override;

        private:
            lvr2::ChunkManager m_chunkManager;
            std::unordered_map<size_t, MeshBufferPtr> m_chunks;
            std::unordered_map<size_t, vtkSmartPointer<vtkActor> > m_chunkActors;

            lvr2::BoundingBox<BaseVector<float> > frustumToBB(double planes[24]);

            double sarrus(const double U[4],
                          const double V[4],
                          const double W[4],
                          size_t A,
                          size_t B,
                          size_t C);
    };
}
