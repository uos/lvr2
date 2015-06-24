#ifndef MARCHING_CUBES_HPP_
#define MARCHING_CUBES_HPP_

#include <reconstruction/FastReconstruction.hpp>
#include <reconstruction/TSDFGrid.hpp>
#include <reconstruction/PointsetSurface.hpp>
#include <reconstruction/FastBox.hpp>
#include <io/PointBuffer.hpp>
#include <io/DataStruct.hpp>
#include <io/Timestamp.hpp>
#include <geometry/HalfEdgeVertex.hpp>
#include <geometry/HalfEdgeMesh.hpp>
#include <geometry/BoundingBox.hpp>
#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include <thread>
#include <queue>


using namespace lvr;

typedef Vertex<float>  fVertex;
typedef ColorVertex<float, unsigned char> cVertex;
typedef FastBox<ColorVertex<float, unsigned char>, lvr::Normal<float> > cFastBox;
typedef TsdfGrid<cVertex, cFastBox, kfusion::Point> TGrid;
typedef FastReconstruction<ColorVertex<float, unsigned char>, lvr::Normal<float>, cFastBox > cFastReconstruction;
typedef HalfEdgeMesh<cVertex, lvr::Normal<float> > HMesh;
typedef HMesh* MeshPtr;

namespace kfusion
{
    class MaCuWrapper
    {
		public:

			MaCuWrapper(double camera_target_distance = 0, double voxel_size = 3.0/512.0);
        
			~MaCuWrapper() 
			{ 
				delete meshPtr_;
				//delete last_grid_;
			}
			
			
			void createGrid(cv::Mat& cloud_host,  Vec3i offset, const bool last_shift);
			
			void createMeshSlice(const bool last_shift);
			
			void optimizeMeshSlice(const bool last_shift);
			
			void fuseMeshSlice(const bool last_shift);
			
			void resetMesh();
			
			MeshPtr getMesh() {return meshPtr_;};
			
			double calcTimeStats();
			
			void setCameraDist(const double threshold) { camera_target_distance_ = threshold;} 
			
			int slice_count_;
			
		private:
		    void transformMeshBack();
		    
			double camera_target_distance_;
		    std::thread* mcthread_;
			queue<TGrid*> grid_queue_;
			queue<MeshPtr> mesh_queue_;
			queue<MeshPtr> opti_mesh_queue_;
			std::vector<double> timeStats_;
			MeshPtr meshPtr_;
			double voxel_size_;
			
    };
}
#endif
