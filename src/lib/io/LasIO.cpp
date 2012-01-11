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
 * LasIO.cpp
 *
 *  @date 03.01.2012
 *  @author Thomas Wiemann
 */

#include <iostream>
using std::cout;
using std::endl;

#include "LasIO.hpp"
#include "io/Timestamp.hpp"

#include "lasreader.hpp"
#include "laswriter.hpp"

namespace lssr
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

        // Read point data
        for(size_t i = 0; i < num_points; i++)
        {
            size_t buf_pos = 3 * i;
            lasreader->read_point();
            points[buf_pos]     = lasreader->point.x;
            points[buf_pos + 1] = lasreader->point.y;
            points[buf_pos + 2] = lasreader->point.z;

            intensities[i] = lasreader->point.intensity;

        }

        // Create point buffer and model
        PointBufferPtr p_buffer( new PointBuffer);
        p_buffer->setPointArray(points, num_points);
        p_buffer->setPointIntensityArray(intensities, num_points);

        ModelPtr m_ptr( new Model(p_buffer));

        delete lasreader;

        return m_ptr;
    }
    else
    {
        cout << timestamp << "LasIO::read(): Unable to open file " << filename << endl;
    }
}


void LasIO::save( string filename )
{
    /// TODO: Implement LAS output
    std::cerr << "LASIO: Saving not yet implemented." << endl;
}

} /* namespace lssr */
