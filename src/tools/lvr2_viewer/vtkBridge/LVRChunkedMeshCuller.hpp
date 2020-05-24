#ifndef LVRCHUNKEDMESHCULLER_HPP_
#define LVRCHUNKEDMESHCULLER_HPP_

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
            ChunkedMeshCuller(LVRChunkedMeshBridge* bridge, double highResDistance = 150.0) : m_bridge(bridge), cull_(true), m_highResDistance(highResDistance){
                std::cout << "Initialized Culler with highResDistance: " << m_highResDistance << std::endl;
            }

            virtual double Cull(vtkRenderer *ren, vtkProp **propList, int &listLength, int &initialized) override;

            void NoCulling() { cull_ = false; }
            void enableCulling() { cull_ = true; }

        private:
            LVRChunkedMeshBridge* m_bridge;
            bool cull_;
            double m_highResDistance;
        };
}

#endif
