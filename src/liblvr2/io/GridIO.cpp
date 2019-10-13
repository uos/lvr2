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
 * GridIO.tcc
 *
 *  @date 10.01.2012
 *  @author Thomas Wiemann
 */

#include "lvr2/io/GridIO.hpp"
#include "lvr2/io/DataStruct.hpp"

#include <fstream>
using std::ifstream;

namespace lvr2
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
        m_boxes = uintArr( new unsigned int[8 * n_cells] );
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

} /* namespace lvr2 */
