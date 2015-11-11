#ifndef MeshStage_HPP__
#define MeshStage_HPP__

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
#include "geometry/Matrix4.hpp"

using namespace lvr;
using namespace std;
using namespace kfusion;

typedef Vertex<float>  fVertex;
typedef ColorVertex<float, unsigned char> cVertex;
typedef FastBox<ColorVertex<float, unsigned char>, lvr::Normal<float> > cFastBox;
typedef TsdfGrid<cVertex, cFastBox, kfusion::Point> TGrid;
typedef FastReconstruction<ColorVertex<float, unsigned char>, lvr::Normal<float>, cFastBox > cFastReconstruction;
typedef HalfEdgeMesh<cVertex, lvr::Normal<float> > HMesh;
typedef HMesh* MeshPtr;

class MeshStage : public AbstractStage
{
public:

	// default constructor
	MeshStage(double camera_target_distance, double voxel_size, Options* options);

	virtual void firstStep();
	virtual void step();
	virtual void lastStep();
	
private:
	void transformMeshBack(MeshPtr mesh);

	queue<MeshPtr> last_mesh_queue_;
	double camera_target_distance_;
	double voxel_size_;
	size_t mesh_count_, fusion_count_;
	Options* options_;
	lvr::Matrix4f global_correction_;
};
#endif // STAGE
