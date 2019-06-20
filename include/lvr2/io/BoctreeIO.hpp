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
 * BoctreeIO.hpp
 *
 *  @date 23.08.2012
 *  @author Thomas Wiemann
 */

#ifndef BOCTREEIO_HPP_
#define BOCTREEIO_HPP_

#include "lvr2/io/BaseIO.hpp"

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"

#include <boost/filesystem.hpp>

#include "slam6d/scan_io_oct.h"


namespace lvr2
{

using Vec = BaseVector<float>;

/**
 * @brief IO-Class to import compressed octrees from slam6d
 */
class BoctreeIO : public BaseIO
{


public:

    BoctreeIO();
    virtual ~BoctreeIO();

    /**
     * \brief Parse the given file and load supported elements.
     *
     * @param filename  The file to read.
     */
    virtual ModelPtr read(string filename );


    /**
     * \brief Save the loaded elements to the given file.
     *
     * @param filename Filename of the file to write.
     */
    virtual void save( string filename );

private:
    Matrix4<Vec> parseFrameFile(ifstream& frameFile);
};

} /* namespace lvr2 */

#endif /* BOCTREEIO_HPP_ */
