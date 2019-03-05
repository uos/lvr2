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
 * @file      DrcIO.hpp
 * @brief     IO module for importing and exporting .drc files
 * @details   Supports geometrys compressed using draco https://github.com/google/draco
 *
 * @author    Steffen Schupp (sschupp), sschupp@uos.de
 * @author	  Malte kl. Piening (mklpiening), mklpiening@uos.de
 *
 **/

#ifndef DRCIO_HPP
#define DRCIO_HPP

#include "BaseIO.hpp"

namespace lvr2
{

class DrcIO : public BaseIO
{
  public:
    DrcIO(){};

    /**
     * @brief Parse the draco and load supported elements.
     *
     * @param filename  The file to read.
     */
    virtual ModelPtr read(string filename);

    /**
     * @brief Save/Compress the loaded elements to a draco file.
     *
     * @param filename Filename of the file to write.
     */
    virtual void save(string filename);

    /**
     * @brief Set the model and saves/compresses the loaded elements
     *  to a draco file
     *
     * @param filename Filename of the file to write.
     */
    virtual void save(ModelPtr model, string filename);
};

} /* namespace lvr */

#endif // DRCIO_HPP