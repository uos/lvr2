#ifndef LVR2_PIPELINE_HPP_
#define LVR2_PIPELINE_HPP_

#include <lvr/reconstruction/FastReconstruction.hpp>
#include <lvr/reconstruction/TSDFGrid.hpp>
#include <lvr/reconstruction/PointsetSurface.hpp>
#include <lvr/reconstruction/FastBox.hpp>
#include <lvr/io/PointBuffer.hpp>
#include <lvr/io/DataStruct.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/geometry/HalfEdgeVertex.hpp>
#include <lvr/geometry/HalfEdgeKinFuMesh.hpp>
#include <lvr/geometry/BoundingBox.hpp>
#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include <kfusion/LinearPipeline.hpp>
#include <kfusion/GridStage.hpp>
#include <kfusion/MeshStage.hpp>
#include <kfusion/OptimizeStage.hpp>
#include <kfusion/FusionStage.hpp>


using namespace lvr;

typedef ColorVertex<float, unsigned char> cVertex;
typedef HalfEdgeKinFuMesh<cVertex, lvr::Normal<float> > HMesh;
typedef HMesh* MeshPtr;

namespace kfusion
{
    class LVRPipeline
    {
		public:

			LVRPipeline(KinFuParams params);

			~LVRPipeline();

			void addTSDFSlice(TSDFSlice slice,  const bool last_shift);

			void resetMesh();

			MeshPtr getMesh() {return pl_.GetResult();}

			double calcTimeStats();

		private:

		    MeshPtr meshPtr_;
		    size_t slice_count_;
			std::vector<double> timeStats_;
			LinearPipeline<pair<TSDFSlice, bool> , MeshPtr> pl_;

    };
}
#endif
