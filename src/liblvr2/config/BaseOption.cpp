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

/*
 * BaseOption.cpp
 *
 *  Created on: Feb 4, 2015
 *      Author: twiemann
 */

#include "lvr2/config/BaseOption.hpp"
#include "lvr2/io/ModelFactory.hpp"

namespace lvr2
{

BaseOption::BaseOption(int argc, char** argv)
	: m_descr("Supported options"), m_argc(argc), m_argv(argv)
{
	m_descr.add_options()
    		("xPos,x", value<int>()->default_value(0), "Position of the x-coordinates in the input point data (according to screen coordinates).")
			("yPos,y", value<int>()->default_value(1), "Position of the y-coordinates in the input data lines (according to screen coordinates).")
			("zPos,z", value<int>()->default_value(2), "Position of the z-coordinates in the input data lines (according to screen coordinates).")
            ("sx", value<float>()->default_value(1.0f), "Scaling factor for the x coordinates.")
            ("sy", value<float>()->default_value(1.0f), "Scaling factor for the y coordinates.")
            ("sz", value<float>()->default_value(1.0f), "Scaling factor for the z coordinates.")
	;
    m_coordinateTransform = new CoordinateTransform<float>;
}

CoordinateTransform<float> BaseOption::coordinateTransform() const
{
	return CoordinateTransform<float>(x(), y(), z(), sx(), sy(), sz());
}

void BaseOption::printTransformation(std::ostream& out) const
{
	out << "##### Program options: " << std::endl;
    if(m_coordinateTransform->transforms())
	{
		out << "##### Transform input data\t: YES" << std::endl;
		out << "##### Position of x coordinates\t: " << x() << std::endl;
		out << "##### Position of y coordinates\t: " << y() << std::endl;
		out << "##### Position of z coordinates\t: " << z() << std::endl;
		out << "##### X-Scale\t\t\t: " << sx() << std::endl;
		out << "##### Y-Scale\t\t\t: " << sy() << std::endl;
		out << "##### Z-Scale\t\t\t: " << sz() << std::endl;
	}
	else
	{
		out << "##### Transform input data\t: NO" << std::endl;
	}
}

void BaseOption::setupInputTransformation()
{

}

void BaseOption::setup()
{
	m_pdescr.add("inputFile", -1);

	// Parse command line and generate variables map
	store(command_line_parser(m_argc, m_argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

	if(sx() != 1.0 || sy() != 1.0 || sz() != 0 || x() != 1 || y() != 1 || z() != 1)
	{
        m_coordinateTransform->x = x();
        m_coordinateTransform->y = y();
        m_coordinateTransform->z = z();
        m_coordinateTransform->sx = sx();
        m_coordinateTransform->sy = sy();
        m_coordinateTransform->sz = sz();

        ModelFactory::m_transform = *m_coordinateTransform;
	}
}

BaseOption::~BaseOption()
{

}

void BaseOption::printLogo() const
{
    std::string logo = R"(
         /\
        /  \               ##          ##      ##    #######         ######
       /    \              ##          ##      ##    ##     ##     ##      ##
      /      \             ##           ##    ##     ##      ##            ##
     /________\            ##           ##    ##     ##     ##            ##
    /\        /\           ##            ##  ##      #######             ##
   /  \      /  \          ##            ##  ##      ##    ##          ##
  /    \    /    \         ##             ####       ##     ##       ##
 /      \  /      \        ##########      ##        ##      ##    ##########
/________\/________\
    )";

    std::cout << logo << std::endl;
}

} /* namespace lvr2 */
