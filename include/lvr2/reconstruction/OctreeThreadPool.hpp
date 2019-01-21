/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*
 * OctreeThreadPool.hpp
 *
 *  Created on: 18.01.2019
 *      Author: Benedikt Schumacher
 */

#ifndef _LVR2_RECONSTRUCTION_OCTREETHREADPOOL_H_
#define _LVR2_RECONSTRUCTION_OCTREETHREADPOOL_H_

#include <boost/thread.hpp>
#include <queue>

using std::vector;
using std::numeric_limits;

namespace lvr2
{

/**
 * @brief A class that implements the thread pool pattern.
 */
template<typename BaseVecT, typename BoxT>
class OctreeThreadPool
{
public:

    /**
     * @brief Constructor.
     *
     * @param number Count of threads.
     */
    OctreeThreadPool(int number);

    /**
     * @brief Starts the tread pool.
     */
    void startPool();

    /**
     * @brief Stops the thread pool.
     */
    void stopPool();

    /**
     * @brief Inserts a task in the queue.
     *
     * @param task A single task
     */
    void insertTask(boost::function<void()> task);

protected:

    /**
     * @brief Assigns a task from the queue to a single thread.
     */
    void work();

    // Count of threads.
    int m_poolSize;

    // Queue with task.
    queue<boost::function<void()>> m_queue;

    // Group with threads.
    boost::thread_group m_threads;

    // Count of threads without a task.
    int m_availableThreads;

    // Mutex to control the threads.
    boost::mutex m_poolMutex;

    // Condition-variable to control the threads.
    boost::condition_variable m_conditionVariable;

    // Status of the activity of the thread pool.
    bool m_isRunning;
};

} // namespace lvr2

#include <lvr2/reconstruction/OctreeThreadPool.tcc>

#endif /* _LVR2_RECONSTRUCTION_OCTREETHREADPOOL_H_ */
