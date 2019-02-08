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
 * @file Hdf5IO.hpp
 */

#ifndef __HDF5_IO_HPP_
#define __HDF5_IO_HPP_


#include <lvr2/io/PlutoMapIO.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <iomanip>
#include <string>
#include <vector>

#include <lvr2/io/Model.hpp>
#include <lvr2/io/BaseIO.hpp>


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

#endif // __HDF5_IO_HPP_
