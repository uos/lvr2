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
 * @file      DracoDecoder.hpp
 * @brief     Decodes a draco comptressed file into a lvr model
 * @details   Supports geometrys compressed using draco https://github.com/google/draco
 *
 * @author    Steffen Schupp (sschupp), sschupp@uos.de
 * @author	  Malte kl. Piening (mklpiening), mklpiening@uos.de
 *
 **/

#ifndef DRACO_DECODER_HPP
#define DRACO_DECODER_HPP

#include "Model.hpp"
#include "draco/compression/decode.h"

namespace lvr2
{

/**
 * @brief delivers ModelPtr from draco DecoderBuffer
 *
 * @param buffer Decoder Buffer thats contents get parsed to a Model
 * @param type GeometryType of loaded structure
 * @return ModelPtr to Model that got created from buffer
 **/
ModelPtr decodeDraco(draco::DecoderBuffer& buffer, draco::EncodedGeometryType type);

} // namespace lvr

#endif