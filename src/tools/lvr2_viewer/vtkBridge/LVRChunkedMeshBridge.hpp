#ifndef LVR2_CHUNKED_MESH_BRIDGE_HPP_
#define LVR2_CHUNKED_MESH_BRIDGE_HPP_

#include <QObject>
//extern "C" {
//#include <GL/glx.h>
//}

#undef Success

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

#include <vtkWeakPointer.h>



//class XContext;
//class XVisualInfo;
//class Window;
//struct XID;

//struct XVisualInfo;
//typedef struct XVisualInfo XVisualInfo;

//struct XVisualInfo;
////typedef struct XVisualInfo XVisualInfo;
//typedef unsigned long XID;
//typedef XID Window;
//struct _XDisplay;
//typedef struct _XDisplay Display;

typedef std::unordered_map<size_t, vtkSmartPointer<vtkActor> > actorMap;
Q_DECLARE_METATYPE(actorMap)

//#include <GL/glx.h>
namespace lvr2 {

template <typename T>
class CompareDistancePair
{

    public:
        bool operator()(std::pair<float,T> p1, std::pair<float,T> p2)
        {
            return p1.first > p2.first;
        }
};


    class LVRChunkedMeshBridge : public QObject
    {
        Q_OBJECT
        public:
            LVRChunkedMeshBridge(std::string file, vtkSmartPointer<vtkRenderer> renderer,
                                 std::vector<std::string> layers, size_t cache_size = 300, double highResDistance = 150.0);
            void getActors(double planes[24],
                    std::vector<BaseVector<float> >& centroids, 
                    std::vector<size_t >& indices);
        
            std::mutex mw_mutex;
            std::condition_variable mw_cond;
            bool release = false;

            std::unordered_map<size_t, vtkSmartPointer<vtkActor>> getHighResActors() { return m_highResActors; }
            std::unordered_map<size_t, vtkSmartPointer<vtkActor>> getLowResActors()  { return m_chunkActors;   }
                    //std::unordered_map<size_t, vtkSmartPointer<vtkActor> >& actors);
            void addInitialActors(vtkSmartPointer<vtkRenderer> renderer);

//            void fetchHighRes(double position[3], double dir[3], double up[3]);
            void fetchHighRes(BoundingBox<BaseVector<float > > bb,
                              std::vector<size_t> indices,
                              std::vector<BaseVector<float>> centroids);
        Q_SIGNALS:
            void updateHighRes(actorMap lowRes, actorMap highRes);
                    
                   // std::unordered_map<size_t, vtkSmartPointer<vtkActor> > lowResActors,
                   //            std::unordered_map<size_t, vtkSmartPointer<vtkActor> > highResActors);

        protected:
            void computeMeshActors();
            inline vtkSmartPointer<vtkActor> computeMeshActor(size_t& id, MeshBufferPtr& chunk);

        private:
            vtkSmartPointer<vtkRenderer> m_renderer;
        //            Display* display;
    //        Window x_window;

            std::thread worker;
            std::mutex mutex;
            std::condition_variable cond_;
            double dist_;
            bool getNew_;
            bool running_;
            BoundingBox<BaseVector<float> > m_region;
            BoundingBox<BaseVector<float> > m_lastRegion;
            // Maybe use 2 maps.
            std::vector<size_t> m_highResIndices;
            std::vector<BaseVector<float> > m_highResCentroids;
            std::vector<size_t > m_lastIndices;
            std::vector<BaseVector<float> > m_lastCentroids;

            std::vector<std::string> m_layers;
            void highResWorker();
            lvr2::ChunkManager m_chunkManager;
            std::unordered_map<size_t, MeshBufferPtr> m_chunks;
            std::unordered_map<size_t, MeshBufferPtr> m_highRes;
            std::unordered_map<size_t, vtkSmartPointer<vtkActor> > m_chunkActors;
//            std::unordered_map<size_t, vtkSmartPointervtkActor> > m_chunkActors;
            std::unordered_map<size_t, vtkSmartPointer<vtkActor> > m_highResActors;

            std::unique_ptr<MeshOctree<BaseVector<float> > > m_oct;
//            std::unordered_map<size_t, std::vector<vtkPolyData> > > m_chunkActors;
    };
} // namespace lvr2

#endif
