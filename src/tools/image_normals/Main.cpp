/**
 * Copyright (C) 2017 Universität Osnabrück
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


// Program options for this tool


#include <lvr/io/ModelFactory.hpp>
#include <lvr/reconstruction/ModelToImage.hpp>

using namespace lvr;


/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{

    ModelPtr model = ModelFactory::readModel(string(argv[1]));

    ModelToImage mti(model->m_pointCloud, ModelToImage::CYLINDRICAL, 6000, 1000, 0, 3000, -360, 360, -90, 90, true, true);

    mti.writePGM("test.pgm", 3000);
}

