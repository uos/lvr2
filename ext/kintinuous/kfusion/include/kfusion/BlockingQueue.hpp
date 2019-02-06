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
 * BlockingQueue.hpp
 *
 *  @date 13.11.2015
 *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
 */

#ifndef BLOCKINGQUEUE_HPP__
#define BLOCKINGQUEUE_HPP__

#include <deque>
#include <limits>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/cstdint.hpp>
#include <boost/any.hpp>

// Blocking Queue based on the Java's implementation.
// This class is thread safe.
class BlockingQueue
{
public:
	enum { NoMaxSizeRestriction = 0 };
	// default constructor
	// provides no size restriction on the blocking queue
	BlockingQueue()
	   : m_maxSize(NoMaxSizeRestriction)
	{
	}

	// Retrieve and remove the oldest element in the queue.
	// If there is no element available, this method will block until
	// elements are inserted.
	//
	// This method is thread safe.
	boost::any Take()
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
	bool Add(boost::any const & o)
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

	// return the maximum allowed size for this queue.
	boost::uint64_t GetMaxSize() const { return m_maxSize; }

	// set the maximum allowed size for this queue.
	void SetMaxSize(boost::uint64_t val) { m_maxSize = val; }

private:

	std::deque<boost::any> m_deque;
	boost::mutex m_mutex;
	boost::uint64_t m_maxSize;
	boost::condition m_condSpaceAvailable;
};
#endif // BLOCKINGQUEUE_HPP__
