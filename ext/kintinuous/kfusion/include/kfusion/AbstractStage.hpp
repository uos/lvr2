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
 * AbstractStage.hpp
 *
 *  @date 13.11.2015
 *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
 */


#ifndef ABSTRACTSTAGE_HPP__
#define ABSTRACTSTAGE_HPP__

#include <kfusion/BlockingQueue.hpp>

// A pipeline stage interface class that can be overridden to perform
// customized steps. Each stage holds pointer to two queues. The in-queue,
// which is the incoming work for this stage, and out-queue, which is the
// completed work from this stage.
class AbstractStage
{
public:

	// default constructor
	AbstractStage()
		: m_done(false)
	{
	}

	void InitQueues( boost::shared_ptr<BlockingQueue > inQueue,
			boost::shared_ptr<BlockingQueue > outQueue)
	{
		m_inQueue = inQueue;
		m_outQueue = outQueue;
	}

	// Activate this stage through this method. FirstStep(), Step() and
	// LastStep() will be invoked sequentially until it's operation is
	// completed.
	void Run()
	{
		firstStep();

		while(done() == false)
		{
			step();
		}

		lastStep();
	}

	// override this method for the first step of this stage
	virtual void firstStep() = 0;
	// override this method for intermediate steps of this stage
	virtual void step() = 0;
	// override this method as the final step of this stage
	virtual void lastStep() = 0;

	boost::shared_ptr<BlockingQueue> getInQueue() const { return m_inQueue; }
	boost::shared_ptr<BlockingQueue> getOutQueue() const { return m_outQueue; }

	bool done() const { return m_done; }
	void done(bool val) { m_done = val; }

protected:

	boost::shared_ptr<BlockingQueue> m_inQueue;
	boost::shared_ptr<BlockingQueue> m_outQueue;

	bool m_done;
};
#endif // ConvertStage_HPP
