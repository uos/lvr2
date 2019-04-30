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
 * QueryPoint.cpp
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

namespace lvr2
{

template<typename BaseVecT>
QueryPoint<BaseVecT>::QueryPoint()
    : QueryPoint(BaseVecT(0.0, 0.0, 0.0))
{}

template<typename BaseVecT>
QueryPoint<BaseVecT>::QueryPoint(const BaseVecT& p)
    : QueryPoint(p, 0.0)
{}

template<typename BaseVecT>
QueryPoint<BaseVecT>::QueryPoint(const BaseVecT& p, float d)
{
    m_position = p;
    m_distance = d;
    m_invalid  = false;
}

template<typename BaseVecT>
QueryPoint<BaseVecT>::QueryPoint(const QueryPoint<BaseVecT> &o)
{
    m_position = o.m_position;
    m_distance = o.m_distance;
    m_invalid  = o.m_invalid;
}


} // namespace lvr
