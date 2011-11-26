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
 * IOFactory.h
 *
 *  @date 24.08.2011
 *  @author Thomas Wiemann
 */

#ifndef IOFACTORY_H_
#define IOFACTORY_H_

#include "Model.hpp"

#include <string>
#include "boost/shared_ptr.hpp"

namespace lssr
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
        static void saveModel( ModelPtr m, std::string file );

};

typedef boost::shared_ptr<ModelFactory> ModelFactoryPtr;

} // namespace lssr

#endif /* IOFACTORY_H_ */
