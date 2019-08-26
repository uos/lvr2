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

namespace lvr2 
{

template <typename BaseVecT>
CudaRaycaster<BaseVecT>::CudaRaycaster(const MeshBufferPtr mesh)
:RaycasterBase<BaseVecT>(mesh)
{
    
}

template <typename BaseVecT>
bool CudaRaycaster<BaseVecT>::castRay(
    const Point<BaseVecT>& origin,
    const Vector<BaseVecT>& direction,
    Point<BaseVecT>& intersection
)
{

    return false;
}

template <typename BaseVecT>
void CudaRaycaster<BaseVecT>::castRays(
    const Point<BaseVecT>& origin,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections,
    std::vector<uint8_t>& hits
)
{
    

}

template <typename BaseVecT>
void CudaRaycaster<BaseVecT>::castRays(
    const std::vector<Point<BaseVecT> >& origins,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections,
    std::vector<uint8_t>& hits
)
{
    
}

} // namespace lvr2