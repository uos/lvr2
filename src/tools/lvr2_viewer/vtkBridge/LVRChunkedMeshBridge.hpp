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

#include <mutex>
#include <thread>
#include <condition_variable>



namespace lvr2 {
    class LVRChunkedMeshBridge  
    {
        public:
            LVRChunkedMeshBridge(std::string file);
            void getActors(double planes[24],
                    std::vector<BaseVector<float> >& centroids, 
                    std::vector<size_t >& indices);
                    
                    //std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> >& actors);
            void addInitialActors(vtkSmartPointer<vtkRenderer> renderer);

            void getHighRes(double x, double y, double z);
        protected:
            void computeMeshActors();
            inline vtkSmartPointer<MeshChunkActor> computeMeshActor(size_t& id, MeshBufferPtr& chunk);

        private:

            std::thread worker;
            std::mutex mutex;
            std::condition_variable cond_;
            double dist_;
            bool getNew_;
            bool running_;
            BoundingBox<BaseVector<float> > region_;

            void highResWorker();
            lvr2::ChunkManager m_chunkManager;
            std::unordered_map<size_t, MeshBufferPtr> m_chunks;
            std::unordered_map<size_t, MeshBufferPtr> m_highRes;
            std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > m_chunkActors;
            std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > m_highResActors;

            MeshOctree<BaseVector<float> >* m_oct;
//            std::unordered_map<size_t, std::vector<vtkPolyData> > > m_chunkActors;
    };
} // namespace lvr2

#endif
