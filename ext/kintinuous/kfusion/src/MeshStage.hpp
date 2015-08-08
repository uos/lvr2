#ifndef GRIDSTAGE_HPP__
#define GRIDSTAGE_HPP__

#include "AbstractStage.hpp"
#include "BlockingQueue.h"
#include <boost/any.hpp>

using namespace std;

class GridStage : public AbstractStage
{
public:

	// default constructor
	GridStage(double voxel_size = 3.0/512.0);

	virtual void FirstStep() { /* omit */ };
	virtual void Step();
	virtual void LastStep()	{ /* omit */ };
	void setLastShift(bool last_shift){last_shift_ = last_shift;}
	
private:
	size_t grid_count_;
	BoundingBox<cVertex> bbox_;
	double voxel_size_;
	queue<TGrid*> last_grid_queue_;
	bool last_shift;
};
#endif // STAGE
