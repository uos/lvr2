#ifndef OptimizeStage_HPP__
#define OptimizeStage_HPP__

#include "AbstractStage.hpp"
#include "BlockingQueue.h"
#include <boost/any.hpp>

using namespace std;

class OptimizeStage : public AbstractStage
{
public:

	// default constructor
	OptimizeStage();

	virtual void firstStep();
	virtual void step();
	virtual void lastStep();
	
private:
	size_t mesh_count_;
};
#endif // STAGE
