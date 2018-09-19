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
#include <lvr2/io/Timestamp.hpp>

#include <iomanip>
#include <string>
#include <vector>

#include <lvr2/io/Model.hpp>
#include <lvr2/io/BaseIO.hpp>

#ifndef __HDF5_IO_HPP_
#define __HDF5_IO_HPP_


using std::string;
using std::vector;

namespace lvr2
{

/**
 * An basic implemntation for the integrated HDF5 format.
 * This saves the mesh geomentry, normals, colors, materials and textures to an
 * defined HDF5 file format.
 *
 * This also tries to mitgate any erros while saving the file to disc, since
 * the HDF5 implementation is very picky about saving anything to an already opened
 * file or any other strange things on the file system. Thus if the first try of
 * saving it the defined filename it tries to make up an new one and saves everything there.
 */
class Hdf5IO : public BaseIO
{
public:
    Hdf5IO() {}
    virtual ~Hdf5IO() {}

    void save(string filename);

    ModelPtr read(string filename);
};

} // namespace lvr2

#include "Hdf5IO.cpp"

#endif // __HDF5_IO_HPP_
