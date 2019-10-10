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
 * Model.h
 *
 *  @date 27.10.2011
 *  @author Thomas Wiemann
 */

#ifndef MODEL_H_
#define MODEL_H_

#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/PointBuffer.hpp"

#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <iostream>

typedef unsigned int uint;

namespace lvr2
{

class Model
{
public:

    Model( PointBufferPtr p = PointBufferPtr(),
           MeshBufferPtr m = MeshBufferPtr() )
    {
        m_pointCloud = p;
        m_mesh = m;
    }

    Model( MeshBufferPtr m, PointBufferPtr p )
    {
        m_pointCloud = p;
        m_mesh = m;
    }

    Model( MeshBufferPtr m )
    {
        m_pointCloud.reset();
        m_mesh = m;
    }

    virtual ~Model() {}

    PointBufferPtr         m_pointCloud;
    MeshBufferPtr          m_mesh;
};

using ModelPtr = std::shared_ptr<Model>;

} // namespace lvr2

#endif /* MODEL_H_ */
