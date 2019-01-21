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
 * OctreeThreadPool.tcc
 *
 *  Created on: 18.01.2019
 *      Author: Benedikt Schumacher
 */

namespace lvr2
{

template<typename BaseVecT, typename BoxT>
OctreeThreadPool<BaseVecT, BoxT>::OctreeThreadPool(int number)
{
    m_poolSize = number;
    m_availableThreads = number;
    m_isRunning = true;
}

template<typename BaseVecT, typename BoxT>
void OctreeThreadPool<BaseVecT, BoxT>::startPool()
{
    for (unsigned char i = 0; i < m_poolSize; ++i)
    {
        m_threads.create_thread(boost::bind(&OctreeThreadPool<BaseVecT, BoxT>::work, this));
    }
}

template<typename BaseVecT, typename BoxT>
void OctreeThreadPool<BaseVecT, BoxT>::stopPool()
{
    while (!(m_queue.empty() && (m_availableThreads == m_poolSize)))
    {
        sleep(0.05);
    }
    boost::unique_lock<boost::mutex> poolLock(m_poolMutex);
    m_isRunning = false;
    poolLock.unlock();
    m_conditionVariable.notify_all();
    m_threads.join_all();
}

template<typename BaseVecT, typename BoxT>
void OctreeThreadPool<BaseVecT, BoxT>::insertTask(boost::function<void()> task)
{
    while (m_availableThreads == 0)
    {
        sleep(0.05);
    }
    boost::unique_lock<boost::mutex> poolLock(m_poolMutex);
    --m_availableThreads;
    m_queue.push(task);
    poolLock.unlock();
    m_conditionVariable.notify_one();
}

template<typename BaseVecT, typename BoxT>
void OctreeThreadPool<BaseVecT, BoxT>::work()
{
    while (m_isRunning)
    {
        boost::unique_lock<boost::mutex> poolLock(m_poolMutex);
        while (m_queue.empty() && m_isRunning)
        {
            m_conditionVariable.wait(poolLock);
        }

        if (!m_isRunning)
        {
            break;
        }

        boost::function<void()> task = m_queue.front();
        m_queue.pop();
        poolLock.unlock();
        task();
        poolLock.lock();
        ++m_availableThreads;
     }
}

} // namespace lvr2
