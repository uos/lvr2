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

/**
 * PointsetSurface.tcc
 *
 *  @date 25.01.2012
 *  @author Thomas Wiemann
 */


namespace lvr2
{

template<typename BaseVecT>
PointsetSurface<BaseVecT>::PointsetSurface(PointBufferPtr pointBuffer)
    : m_pointBuffer(pointBuffer)
{
    // Calculate bounding box
    auto numPoints = m_pointBuffer->numPoints();
    floatArr pts = m_pointBuffer->getPointArray();

    for(size_t i = 0; i < numPoints; i++)
    {
        this->m_boundingBox.expand(BaseVecT(pts[i*3 + 0], pts[i*3 + 1], pts[i*3 + 2]));
    }
}

template<typename BaseVecT>
Normal<float> PointsetSurface<BaseVecT>::getInterpolatedNormal(const BaseVecT& position) const
{
    FloatChannelOptional normals = m_pointBuffer->getFloatChannel("normals"); 
    vector<size_t> indices;
    Normal<float> result;
    m_searchTree->kSearch(position, m_ki, indices);
    for (int i = 0; i < m_ki; i++)
    {
        Normal<float> n = (*normals)[indices[i]];
        result += n;
    }
    result /= m_ki;
    return Normal<float>(result);
}

template<typename BaseVecT>
std::shared_ptr<SearchTree<BaseVecT>> PointsetSurface<BaseVecT>::searchTree() const
{
    return m_searchTree;
}

template<typename BaseVecT>
const BoundingBox<BaseVecT>& PointsetSurface<BaseVecT>::getBoundingBox() const
{
    return m_boundingBox;
}

template<typename BaseVecT>
PointBufferPtr PointsetSurface<BaseVecT>::pointBuffer() const
{
    return m_pointBuffer;
}

} // namespace lvr2
