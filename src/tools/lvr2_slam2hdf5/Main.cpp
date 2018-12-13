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


#include "Options.hpp"
#include <lvr2/io/ScanData.hpp>
#include <lvr2/io/HDF5IO.hpp>

using namespace lvr2;

int main(int argc, char** argv)
{
    // Parse command line arguments
    slam2hdf5::Options options(argc, argv);

    vector<ScanData> scans;
    parseSLAMDirectory(options.inputDir(), scans);

    HDF5IO hdf5("test.h5", true);

    for(size_t i = 0; i < scans.size(); i++)
    {
        size_t reduction_factor = 20;
        size_t num_pts_preview = scans[i].m_points->numPoints() / reduction_factor;
        floatArr pts_preview = floatArr( new float[num_pts_preview * 3 + 3] );
        floatArr points = scans[i].m_points->getPointArray();
        size_t preview_idx = 0;
        for (size_t j = 0; j < scans[i].m_points->numPoints(); j++)
        {
            if (j % reduction_factor == 0)
            {
                pts_preview[preview_idx*3 + 0] = points[j*3 + 0];
                pts_preview[preview_idx*3 + 1] = points[j*3 + 1];
                pts_preview[preview_idx*3 + 2] = points[j*3 + 2];
                preview_idx++;
            }
        }
        scans[i].m_preview = PointBufferPtr( new PointBuffer(pts_preview, num_pts_preview) );
        hdf5.addRawScanData((int)i, scans[i]);
    }

	return 0;
}

