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
 * Model.h
 *
 *  @date 27.10.2011
 *  @author Thomas Wiemann
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/PointBuffer2.hpp>

#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <iostream>

typedef unsigned int uint;

namespace lvr2
{

template<typename BaseVecT>
class Model
{
public:

    Model( PointBuffer2Ptr p = PointBuffer2Ptr(),
           MeshBufferPtr<BaseVecT> m = MeshBufferPtr<BaseVecT>() )
    {
        m_pointCloud = p;
        m_mesh = m;
    }

    Model( MeshBufferPtr<BaseVecT> m, PointBuffer2Ptr p )
    {
        m_pointCloud = p;
        m_mesh = m;
    }

    Model( MeshBufferPtr<BaseVecT> m )
    {
        m_pointCloud.reset();
        m_mesh = m;
    }

    virtual ~Model() {}

    PointBuffer2Ptr             m_pointCloud;
    MeshBufferPtr<BaseVecT>     m_mesh;
};

template<typename T>
using ModelPtr = std::shared_ptr<Model<T>>;

} // namespace lvr

#endif /* MODEL_H_ */
