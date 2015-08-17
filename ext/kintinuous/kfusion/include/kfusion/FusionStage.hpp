#ifndef FusionStage_HPP__
#define FusionStage_HPP__

#include "AbstractStage.hpp"
#include "BlockingQueue.hpp"
#include <boost/any.hpp>
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


class FusionStage : public AbstractStage
{
public:

	// default constructor
	FusionStage(MeshPtr mesh, double camera_target_distance_, double voxel_size);

	virtual void firstStep();
	virtual void step();
	virtual void lastStep();
	
private:
	void transformMeshBack();
	
	size_t mesh_count_;
	MeshPtr mesh_;
	double camera_target_distance_;
	double voxel_size_;
};
#endif // STAGE
