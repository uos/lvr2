#ifndef ABSTRACTBLOCKINGQUEUE_HPP__
#define ABSTRACTBLOCKINGQUEUE_HPP__

#include <deque>
#include <limits>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/cstdint.hpp>
#include <boost/any.hpp>

// Blocking Queue based on the Java's implementation.
// This class is thread safe.
class AbstractBlockingQueue
{
public:
	enum { NoMaxSizeRestriction = 0 };

	// default constructor
	// provides no size restriction on the blocking queue
	AbstractBlockingQueue()
		: m_maxSize(NoMaxSizeRestriction)
	{
	}

	// Retrieve and remove the oldest element in the queue. 
	// If there is no element available, this method will block until
	// elements are inserted.
	//
	// This method is thread safe.
	virtual boost::any Take() = 0;
	// Insert a new element to the blocking queue. 
	// Return true if the operation is successful, false otherwise.
	//
	// This method is thread safe.
	virtual bool Add(boost::any const & o) = 0;

	// return the maximum allowed size for this queue.
	boost::uint64_t GetMaxSize() const { return m_maxSize; }

	// set the maximum allowed size for this queue.
	void SetMaxSize(boost::uint64_t val) { m_maxSize = val; }
protected:
	boost::uint64_t m_maxSize;
};
#endif // BLOCKINGQUEUE_HPP__
