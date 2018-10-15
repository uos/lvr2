/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
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
#include <lvr2/geometry/Vector.hpp>

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
    
    void getPointPairs(PointPairVector<BaseVecT>& pairs, Vector<BaseVecT>& centroid_m, Vector<BaseVecT>& centroid_d, double& sum);

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
