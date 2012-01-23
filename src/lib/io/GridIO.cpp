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

void GridIO::read( std::string filename )
{
    ifstream in(filename.c_str());

    if(in.good())
    {
        size_t n_points;
        size_t n_cells;
        float voxelsize;

        // Read header
        in >> n_points >> voxelsize >> n_cells;

        // Alloc and read points
        m_points = floatArr( new float[4 * n_points] );
        m_numPoints = n_points;
        for(size_t i = 0; i < n_points; i++)
        {
            in >> m_points[i * 4] >> m_points[i * 4 + 1] >> m_points[i * 4 + 2] >> m_points[i * 4 + 3];
        }

        // Alloc and read box indices
        m_boxes = uintArr( new uint[8 * n_cells] );
        m_numBoxes = n_cells;
        for(size_t i = 0; i < n_cells; i++)
        {
            size_t pos = i * 8;
            for(size_t j = 0; j < 8; j++)
            {
                in >> m_boxes[pos + j];
            }
        }

    }
}


floatArr GridIO::getPoints( size_t &n )
{
    n = m_numPoints;
    return m_points;
}


uintArr GridIO::getBoxes( size_t &n )
{
    n = m_numBoxes;
    return m_boxes;
}


GridIO::~GridIO()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lssr */
