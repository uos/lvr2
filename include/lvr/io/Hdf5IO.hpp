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
 * @file Hdf5IO.hpp
 */

#include <lvr2/io/PlutoMapIO.hpp>
#include <lvr/io/Timestamp.hpp>

#include <iomanip>
#include <string>
#include <vector>

#include "Model.hpp"
#include "BaseIO.hpp"

#ifndef __HDF5_IO_HPP_
#define __HDF5_IO_HPP_


using std::string;
using std::vector;

namespace lvr
{

/**
 *
 */
class Hdf5IO : public BaseIO
{
public:
    Hdf5IO() {}
    virtual ~Hdf5IO() {}

    void save(string filename);

    ModelPtr read(string filename);
};

}

#endif // __HDF5_IO_HPP_
