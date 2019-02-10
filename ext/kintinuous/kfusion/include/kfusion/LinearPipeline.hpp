/*
 * Software License Agreement (BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
/*
 * LinearPipeline.hpp
 *
 *  @date 13.11.2015
 *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
 */

#ifndef LINEAR_PIPELINE_HPP__
#define LINEAR_PIPELINE_HPP__

#include "BlockingQueue.hpp"
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
			m_queues[numStage-1] =
				boost::shared_ptr<BlockingQueue >(
					new BlockingQueue()
					);
		}
		// allocate a queue for the new stage
		m_queues[numStage] =
			boost::shared_ptr<BlockingQueue >(
				new BlockingQueue()
				);

		// initialize the stage with the in and out queue
		m_stages[numStage-1]->InitQueues(
			m_queues[numStage-1], m_queues[numStage]
			);
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
	void join()
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
		boost::shared_ptr<BlockingQueue >
	> m_queues;

	std::vector<
		boost::shared_ptr<boost::thread>
	> m_threads;
};
#endif // LinearPipeline_h__
