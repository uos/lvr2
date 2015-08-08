#ifndef BLOCKINGQUEUE_HPP__
#define BLOCKINGQUEUE_HPP__

#include "AbstractBlockingQueue.hpp"
#include <deque>
#include <limits>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/cstdint.hpp>

// Blocking Queue based on the Java's implementation.
// This class is thread safe.
class BlockingQueue : public AbstractBlockingQueue
{
public:

	// default constructor
	// provides no size restriction on the blocking queue
	BlockingQueue()
		: AbstractBlockingQueue()
	{
	}

	// Retrieve and remove the oldest element in the queue. 
	// If there is no element available, this method will block until
	// elements are inserted.
	//
	// This method is thread safe.
	virtual boost::any Take()
	{
		boost::mutex::scoped_lock guard(m_mutex);

		if(m_deque.empty() == true)
		{
			m_condSpaceAvailable.wait(guard);
		}

		boost::any o = m_deque.back();

		m_deque.pop_back();

		return o;
	}
	// Insert a new element to the blocking queue. 
	// Return true if the operation is successful, false otherwise.
	//
	// This method is thread safe.
	virtual bool Add(boost::any const & o)
	{
		boost::mutex::scoped_lock guard(m_mutex);

		if( (m_deque.size() >= m_maxSize) && 
			(NoMaxSizeRestriction!= m_maxSize) )
		{
			return false;
		}

		m_deque.push_front(o);

		m_condSpaceAvailable.notify_one();

		return true;
	}

private:

	std::deque<boost::any> m_deque;
	boost::mutex m_mutex;
	boost::condition m_condSpaceAvailable;
};
#endif // BLOCKINGQUEUE_HPP__
