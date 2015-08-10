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
		m_inQueue = boost::shared_ptr<BlockingQueue>(
			new BlockingQueue());
		m_outQueue = boost::shared_ptr<BlockingQueue>(
			new BlockingQueue());
	}

	// Activate this stage through this method. FirstStep(), Step() and
	// LastStep() will be invoked sequentially until it's operation is 
	// completed.
	void Run()
	{
		FirstStep();

		while(done() == false)
		{
			Step();
		}

		LastStep();
	}

	// override this method for the first step of this stage
	virtual void FirstStep() = 0;
	// override this method for intermediate steps of this stage
	virtual void Step() = 0;
	// override this method as the final step of this stage
	virtual void LastStep() = 0;

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
