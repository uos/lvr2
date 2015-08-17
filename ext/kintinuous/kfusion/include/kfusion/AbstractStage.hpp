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
