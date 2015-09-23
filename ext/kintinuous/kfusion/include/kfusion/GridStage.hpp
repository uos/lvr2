#ifndef GRIDSTAGE_HPP__
#define GRIDSTAGE_HPP__

#include "AbstractStage.hpp"
#include "BlockingQueue.hpp"
#include <boost/any.hpp>
#include <reconstruction/FastReconstruction.hpp>
#include <reconstruction/TSDFGrid.hpp>
#include <reconstruction/PointsetSurface.hpp>
#include <reconstruction/FastBox.hpp>
#include <io/PointBuffer.hpp>
#include <io/DataStruct.hpp>
#include <geometry/HalfEdgeVertex.hpp>
#include <geometry/HalfEdgeMesh.hpp>
#include <geometry/BoundingBox.hpp>
#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/projective_icp.hpp>

using namespace lvr;
using namespace kfusion;
using namespace std;

typedef Vertex<float>  fVertex;
typedef ColorVertex<float, unsigned char> cVertex;
typedef FastBox<ColorVertex<float, unsigned char>, lvr::Normal<float> > cFastBox;
typedef TsdfGrid<cVertex, cFastBox, kfusion::Point> TGrid;
typedef FastReconstruction<ColorVertex<float, unsigned char>, lvr::Normal<float>, cFastBox > cFastReconstruction;
typedef HalfEdgeMesh<cVertex, lvr::Normal<float> > HMesh;
typedef HMesh* MeshPtr;

class GridStage : public AbstractStage
{
public:

	// default constructor
	GridStage(double voxel_size = 3.0/512.0, Options* options = NULL);

	void firstStep();
	void step();
	void lastStep();
	
private:
	size_t grid_count_;
	BoundingBox<cVertex> bbox_;
	double voxel_size_;
	queue<TGrid*> last_grid_queue_;
	bool last_shift;
};
#endif // STAGE
