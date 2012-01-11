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
 * GridIO.tcc
 *
 *  @date 10.01.2012
 *  @author Thomas Wiemann
 */

#include "GridIO.hpp"

#include <fstream>
using std::ifstream;

namespace lssr
{

GridIO::GridIO()
{
    // TODO Auto-generated constructor stub

}

Grid* GridIO::read(string filename)
{
    ifstream in(filename.c_str());

    if(in.good())
    {
        int n_points;
        int n_cells;
        float voxelsize;

        // Read header
        in >> n_points >> voxelsize >> n_cells;

        // Alloc and read points
        floatArr points(new float[4 * n_points]);
        for(int i = 0; i < n_points; i++)
        {
            in >> points[i * 4] >> points[i * 4 + 1] >> points[i * 4 + 2] >> points[i * 4 + 3];
        }

        // Alloc and read box indices
        uintArr boxes(new uint[8 * n_cells]);
        for(int i = 0; i < n_cells; i++)
        {
            int pos = i * 8;
            for(int j = 0; j < 8; j++)
            {
                in >> boxes[pos + j];
            }
        }

        return new Grid(points, boxes, n_points, n_cells);
    }
    else
    {
        return 0;
    }
}


GridIO::~GridIO()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lssr */
