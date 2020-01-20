#include <vtkCuller.h>

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>

#include "LVRChunkedMeshBridge.hpp"

#include <string>




namespace lvr2 {
    class ChunkedMeshCuller: public vtkCuller
    {
        public:
            ChunkedMeshCuller(LVRChunkedMeshBridge* bridge) : m_bridge(bridge) {}

            virtual double Cull(vtkRenderer *ren, vtkProp **propList, int &listLength, int &initialized) override;

        private:
            LVRChunkedMeshBridge* m_bridge;
            lvr2::BoundingBox<BaseVector<float> > frustumToBB(double planes[24]);

            double sarrus(const double U[4],
                          const double V[4],
                          const double W[4],
                          size_t A,
                          size_t B,
                          size_t C);
    };
}
