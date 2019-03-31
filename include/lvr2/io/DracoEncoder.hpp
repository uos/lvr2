/* Copyright (C) 2018 Uni Osnabrück
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
 *
 * @file      DracoEncoder.hpp
 * @brief     Encodes a lvr model into a draco compressed file
 * @details   Supports geometrys compressed using draco https://github.com/google/draco
 *
 * @author    Steffen Schupp (sschupp), sschupp@uos.de
 * @author	  Malte kl. Piening (mklpiening), mklpiening@uos.de
 *
 **/

#ifndef DRACO_ENCODER_HPP
#define DRACO_ENCODER_HPP

#include "Model.hpp"
#include "draco/compression/encode.h"

namespace lvr2
{

/**
 * @brief encodes Model to draco EncodeBuffer which contents can be written into a file
 *
 * @param modelptr modelPtr to Model that shall be encoded
 * @param type GeometryType of Geometry to be encoded
 *
 * @return unique_ptr pointing to a EncoderBuffer that can be used to write a draco file
 **/
std::unique_ptr<draco::EncoderBuffer> encodeDraco(ModelPtr                   modelPtr,
                                                  draco::EncodedGeometryType type);

} // namespace lvr

#endif