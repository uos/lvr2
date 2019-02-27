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
 * ICPPointAlign.hpp
 *
 *  @date Mar 18, 2014
 *  @author Thomas Wiemann
 */
#ifndef ICPPOINTALIGN_HPP_
#define ICPPOINTALIGN_HPP_

#include <lvr2/registration/EigenSVDPointAlign.hpp>
#include <lvr2/reconstruction/SearchTree.hpp>
#include <lvr2/geometry/Matrix4.hpp>

namespace lvr2
{

template <typename BaseVecT>
class ICPPointAlign
{
public:
    ICPPointAlign(PointBufferPtr model, PointBufferPtr data, Matrix4<BaseVecT> transformation);

    Matrix4<BaseVecT> match();

    virtual ~ICPPointAlign();

    void    setMaxMatchDistance(double distance);
    void    setMaxIterations(int iterations);
    void    setEpsilon(double epsilon);

    double  getEpsilon();
    double  getMaxMatchDistance();
    int     getMaxIterations();
    
    void getPointPairs(PointPairVector<BaseVecT>& pairs, BaseVecT& centroid_m, BaseVecT& centroid_d, double& sum);

protected:

    void transform();

    double                              m_epsilon;
    double                              m_maxDistanceMatch;
    int                                 m_maxIterations;

    PointBufferPtr                     m_modelCloud;
    PointBufferPtr                     m_dataCloud;
    Matrix4<BaseVecT>                   m_transformation;

    SearchTreePtr<BaseVecT> 			m_searchTree;
};

} /* namespace lvr2 */

#include <lvr2/registration/ICPPointAlign.tcc>

#endif /* ICPPOINTALIGN_HPP_ */
