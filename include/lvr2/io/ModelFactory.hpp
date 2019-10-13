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
 * IOFactory.h
 *
 *  @date 24.08.2011
 *  @author Thomas Wiemann
 */

#ifndef IOFACTORY_H_
#define IOFACTORY_H_

#include "lvr2/io/Model.hpp"
#include "lvr2/io/CoordinateTransform.hpp"

#include <string>
#include <vector>
#include <map>

#include <boost/shared_ptr.hpp>


namespace lvr2
{

/**
 * @brief Factory class extract point cloud and mesh information
 *        from supported file formats. The instantiated MeshLoader
 *        and PointLoader instances are persistent, i.e. they will
 *        not be freed in the destructor of this class to prevent
 *        side effects.
 */
class ModelFactory
{
    public:

        static ModelPtr readModel( std::string filename );

        static void saveModel( ModelPtr m, std::string file);

        static CoordinateTransform<float> m_transform;

};

typedef boost::shared_ptr<ModelFactory> ModelFactoryPtr;

} // namespace lvr2

#endif /* IOFACTORY_H_ */
