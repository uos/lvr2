#ifndef ConvertStage_HPP__
#define ConvertStage_HPP__

#include "BlockingQueue.h"

// A pipeline stage interface class that can be overridden to perform
// customized steps. Each stage holds pointer to two queues. The in-queue,
// which is the incoming work for this stage, and out-queue, which is the
// completed work from this stage.
template<typename WorkTypeIN, WorktypeOUT>
class Stage
{
public:

	// default constructor
	ConvertStage()
		: m_done(false)
	{		
	}

	// initialize the incoming queue and outgoing queue
	void InitQueues(
		boost::shared_ptr<BlockingQueue<WorkTypeIN> > inQueue, 
		boost::shared_ptr<BlockingQueue<WorkTypeOUT> > outQueue)

	{
		m_inQueue = inQueue;
		m_outQueue = outQueue;
	}

	// Activate this stage through this method. FirstStep(), Step() and
	// LastStep() will be invoked sequentially until it's operation is 
	// completed.
	void Run()
	{
		FirstStep();

		while(Done() == false)
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

	BlockingQueue<WorkTypeIN> & GetInQueue() const { return *m_inQueue; }
	BlockingQueue<WorkTypeOUT> & GetOutQueue() const { return *m_outQueue; }

	bool Done() const { return m_done; }
	void Done(bool val) { m_done = val; }	

private:

	boost::shared_ptr<BlockingQueue<WorkTypeIN> > m_inQueue;
	boost::shared_ptr<BlockingQueue<WorkTypeOUT> > m_outQueue;

	bool m_done;
};
#endif // ConvertStage_HPP
