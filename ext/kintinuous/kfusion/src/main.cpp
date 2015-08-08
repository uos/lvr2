#include <iostream>
#include "LinearPipeline.h"
#include "StreamStage.hpp"

int main()
{
	// create a linear pipeline that takes vector of integer as work unit
	LinearPipeline<std::vector<int>,  std::vector<int> > pl;

	// add a add-one stage
	pl.AddStage(
		boost::shared_ptr<GridStage>(new GridStage())
		);

	// add a add-one stage
	/*pl.AddStage(
		boost::shared_ptr<CAddOneStage>(new CAddOneStage())
		);

	// add a add-one stage
	pl.AddStage(
		boost::shared_ptr<CAddOneStage>(new CAddOneStage())
		);*/

	// at this point, the pipeline has three stages that adds one to 
	// a vector

	// start the pipeline stages by spinning up all the threads
	pl.Start();

	// create a work unit
	std::vector<int> work(100,0);

	// add in some work
	for(size_t i=0; i<10; ++i)
	{
		pl.AddWork(work);
	}

	// extract the results
	for(size_t i=0; i<10; ++i)
	{
		std::vector<int> result = pl.GetResult();
		std::cout << result[0] << std::endl;
	}

	// wait for all the pipeline to complete
	// currently, the pipeline will never complete because the stages
	// don't know when to finish. An API can be implemented easily to end
	// the pipeline no matter what.
	pl.Join();

	return 0;
}
