#ifndef LINEAR_PIPELINE_HPP__
#define LINEAR_PIPELINE_HPP__

#include "BlockingQueue.h"
#include "AbstractStage.hpp"
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

// A class that represents a pipeline that holds numerous stages, and
// each stage can be executed concurrently.
template<typename WorkTypeA, typename WorkTypeB>
class LinearPipeline
{
public:

	// Add a stage to the pipeline. The stages are inserted in order, such
	// that the first added stage will be the first stage, and so on. The
	// work queues for each stage will be updated automatically.
	void AddStage(boost::shared_ptr<AbstractStage > stage)
	{
		// add a stage
		m_stages.push_back(stage);
		size_t numStage = m_stages.size();
		
		// resize the queue accordingly
		m_queues.resize(numStage+1);

		// special case for the first stage, where queue[0] needs to
		// be allocated.
		if(m_queues[numStage-1] == 0)
		{
			m_queues[numStage-1] = stage->getInQueue();
		}
		// allocate a queue for the new stage
		m_queues[numStage] = stage->getOutQueue();
	}


	// Add work to the first queue, which is the in-queue for the first
	// stage.
	void AddWork(WorkTypeA work)
	{
		m_queues[0]->Add(work);
	}

	// Extract the result from the out-queue of the last stage
	WorkTypeB GetResult()
	{
		return boost::any_cast<WorkTypeB>(m_queues[m_queues.size()-1]->Take());
	}

	// Start all stages by spinning up one thread per stage.
	void Start()
	{
		for(size_t i=0; i<m_stages.size(); ++i)
		{
			m_threads.push_back(
				boost::shared_ptr<boost::thread>(new boost::thread(
				boost::bind(&LinearPipeline<WorkTypeA, WorkTypeB>::StartStage, this, i)
				)));
		}
	}

	// join all stages 
	void Join()
	{
		for(size_t i=0; i<m_stages.size(); ++i)
		{
			m_threads[i]->join();
		}
	}

private:

	void StartStage(size_t index)
	{
		m_stages[index]->Run();
	}

	std::vector<
		boost::shared_ptr<AbstractStage > 
	> m_stages;

	std::vector<
		boost::shared_ptr<AbstractBlockingQueue > 
	> m_queues;

	std::vector<
		boost::shared_ptr<boost::thread> 
	> m_threads;
};
#endif // LinearPipeline_h__
