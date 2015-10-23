#ifndef MARCHING_CUBES_HPP_
#define MARCHING_CUBES_HPP_

#include <reconstruction/FastReconstruction.hpp>
#include <reconstruction/TSDFGrid.hpp>
#include <reconstruction/PointsetSurface.hpp>
#include <reconstruction/FastBox.hpp>
#include <io/PointBuffer.hpp>
#include <io/DataStruct.hpp>
#include <io/Timestamp.hpp>
#include <geometry/HalfEdgeMesh.hpp>
#include <geometry/BoundingBox.hpp>
#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/projective_icp.hpp>


using namespace lvr;

typedef Vertex<float>  fVertex;
typedef ColorVertex<float, unsigned char> cVertex;
typedef FastBox<ColorVertex<float, unsigned char>, lvr::Normal<float> > cFastBox;
typedef TsdfGrid<cVertex, cFastBox, kfusion::Point> TGrid;
typedef FastReconstruction<ColorVertex<float, unsigned char>, lvr::Normal<float>, cFastBox > cFastReconstruction;

namespace kfusion
{
    class MaCuWrapper
    {
		public:

			MaCuWrapper();
        
			~MaCuWrapper() 
			{ 
				delete meshPtr_;
				delete last_grid_;
			}
			
			void createMeshSlice(cv::Mat& cloud_host,  Vec3i offset, const bool last_shift);
			
			void resetMesh();
			
			double calcTimeStats();
			
			int slice_count_;
        
		private:
			TGrid* last_grid_;
			std::vector<double> timeStats_;
			HalfEdgeMesh<ColorVertex<float, unsigned char> , lvr::Normal<float> > *meshPtr_;
			
    };
}
#endif
