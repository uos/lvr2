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
 * LasIO.cpp
 *
 *  @date 03.01.2012
 *  @author Thomas Wiemann
 */

#include <iostream>
using std::cout;
using std::endl;

#include "lvr2/io/LasIO.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <lasreader.hpp>
#include <laswriter.hpp>

namespace lvr2
{

ModelPtr LasIO::read(string filename )
{

    // Create Lasreader object
    LASreadOpener lasreadopener;
    lasreadopener.set_file_name(filename.c_str());

    if(lasreadopener.active())
    {
        LASreader* lasreader = lasreadopener.open();

        // Get number of points in file
        size_t num_points = lasreader->npoints;

        // Alloc coordinate array
        floatArr points ( new float[3 * num_points]);
        floatArr intensities ( new float[num_points]);
        ucharArr colors (new unsigned char[3 * num_points]);

        // Read point data
        for(size_t i = 0; i < num_points; i++)
        {
            size_t buf_pos = 3 * i;
            lasreader->read_point();
            points[buf_pos]     = lasreader->point.x;
            points[buf_pos + 1] = lasreader->point.y;
            points[buf_pos + 2] = lasreader->point.z;

            // Create fake colors from intensities
            /// TODO: Check for color attributes if possible...
            colors[buf_pos] = lasreader->point.intensity;
            colors[buf_pos + 1] = lasreader->point.intensity;
            colors[buf_pos + 2] = lasreader->point.intensity;

            intensities[i] = lasreader->point.intensity;

        }

        // Create point buffer and model
        PointBufferPtr p_buffer( new PointBuffer);
        p_buffer->setPointArray(points, num_points);
        p_buffer->addFloatChannel(intensities, "intensities", num_points, 1);
        p_buffer->setColorArray(colors, num_points);

        ModelPtr m_ptr( new Model(p_buffer));
        m_model = m_ptr;

        delete lasreader;

        return m_ptr;
    }
    else
    {
        cout << timestamp << "LasIO::read(): Unable to open file " << filename << endl;
        return ModelPtr();
    }
}


void LasIO::save( string filename )
{
    /// TODO: Implement LAS output
    std::cerr << "LASIO: Saving not yet implemented." << endl;
}

} /* namespace lvr2 */
