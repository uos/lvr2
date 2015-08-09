#ifndef FusionStage_HPP__
#define FusionStage_HPP__

#include "AbstractStage.hpp"
#include "BlockingQueue.h"
#include <boost/any.hpp>

using namespace std;

class FusionStage : public AbstractStage
{
public:

	// default constructor
	FusionStage(MeshPtr mesh);

	virtual void firstStep();
	virtual void step();
	virtual void lastStep();
	
private:
	size_t mesh_count_;
	MeshPtr mesh_;
};
#endif // STAGE
