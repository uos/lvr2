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
 * ICPPointAlign.cpp
 *
 *  @date Mar 18, 2014
 *  @author Thomas Wiemann
 */
#include "registration/ICPPointAlign.hpp"

namespace lvr
{

ICPPointAlign::ICPPointAlign(PointBufferPtr model, PointBufferPtr data) :
    m_modelCloud(model), m_dataCloud(data)
{
    // Init default values
    m_epsilon               = 0.00001;
    m_maxDistanceMatch      = 25;
    m_maxIterations         = 50;

    size_t numPoints = model->getNumPoints();

    // Create search tree
    m_searchTree = SearchTreeFlann<Vertexf>::Ptr(new SearchTreeFlann<Vertexf>(model, numPoints));
}

ICPPointAlign::~ICPPointAlign()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
