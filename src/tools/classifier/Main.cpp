/**
 * Copyright (C) 2013 Universität Osnabrück
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
 * @file Main.cpp
 * @author Simon Herkenhoff <sherkenh@uni-osnabrueck.de>
 */


// Program options for this tool

#ifndef DEBUG
  #include "Options.hpp"
#endif

// Local includes
#include "classifier/AdaptiveKSearchSurface.hpp"
#include "classification/ClassifierFactory.hpp"
#include "classification/ColorGradientPlaneClassifier.hpp"
#include "classification/IndoorNormalClassifier.hpp"
#include "classification/RegionClassifier.hpp"
#include "io/PLYIO.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/HalfEdgeMesh.hpp"
#include "texture/Texture.hpp"
#include "texture/Transform.hpp"
#include "texture/Texturizer.hpp"
#include "texture/Statistics.hpp"
#include "geometry/QuadricVertexCosts.hpp"
#include "reconstruction/SharpBox.hpp"

#include <iostream>

using namespace lvr;

/**
 * @brief Main entry point for the LVR classifier executable
 */
int main(int argc, char** argv)
{
	return 0;
}

