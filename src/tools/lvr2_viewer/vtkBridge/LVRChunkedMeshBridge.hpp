#ifndef LVR2_CHUNKED_MESH_BRIDGE_HPP_
#define LVR2_CHUNKED_MESH_BRIDGE_HPP_

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>

#include "lvr2/algorithm/ChunkManager.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/display/MeshOctree.hpp"

#include <string>

#include "MeshChunkActor.hpp"



namespace lvr2 {
    class LVRChunkedMeshBridge  
    {
        public:
            LVRChunkedMeshBridge(std::string file);
            void getActors(double planes[24],
                    std::vector<size_t>& indices);
                    
                    //std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> >& actors);
            void addInitialActors(vtkSmartPointer<vtkRenderer> renderer);
        protected:
            void computeMeshActors();

        private:
            lvr2::ChunkManager m_chunkManager;
            std::unordered_map<size_t, MeshBufferPtr> m_chunks;
            std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > m_chunkActors;
            MeshOctree<BaseVector<float> >* m_oct;
//            std::unordered_map<size_t, std::vector<vtkPolyData> > > m_chunkActors;
    };
} // namespace lvr2

#endif
