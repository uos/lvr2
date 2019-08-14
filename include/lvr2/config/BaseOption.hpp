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
 * BaseOption.hpp
 *
 *  Created on: Feb 4, 2015
 *      Author: twiemann
 */

#ifndef INCLUDE_LIBLVR2_CONFIG_BASEOPTION_HPP_
#define INCLUDE_LIBLVR2_CONFIG_BASEOPTION_HPP_

#include <boost/program_options.hpp>
#include <iostream>

#include "lvr2/io/CoordinateTransform.hpp"

namespace lvr2
{

using namespace boost::program_options;

class BaseOption
{
public:
	BaseOption(int argc, char** argv);
	virtual ~BaseOption();

    /**
     * @brief 	Prints transformation information to the given output stream
     */
    void printTransformation(std::ostream& out) const;

	/**
	 * @brief   Returns the scaling factor for the x coordinates
	 */
	float sx() const { return m_variables["sx"].as<float>();}

	/**
	 * @brief   Returns the scaling factor for the y coordinates
	 */
	float sy() const { return m_variables["sy"].as<float>();}

	/**
	 * @brief   Returns the scaling factor for the z coordinates
	 */
	float sz() const { return m_variables["sz"].as<float>();}

	/**
	 * @brief   Returns the position of the x coordinate in the data.
	 */
	int x() const { return m_variables["xPos"].as<int>();}

	/**
	 * @brief   Returns the position of the x coordinate in the data.
	 */
	int y() const { return m_variables["yPos"].as<int>();}

	/**
	 * @brief   Returns the position of the x coordinate in the data.
	 */
	int z() const { return m_variables["zPos"].as<int>();}

	/// Returns the coordinate system transformation stored
	/// in the option
	CoordinateTransform<float> coordinateTransform() const;

    void printLogo() const;

protected:

	/// Setup internal data structures
	virtual void setup();

    /// Setup transformation info for ModelIO
    void setupInputTransformation();

	/// The internally used variable map
	variables_map			        m_variables;

	/// The internally used option description
	options_description 		    m_descr;

	/// The internally used positional option desription
	positional_options_description 	m_pdescr;

    /// Coordinate transform information
    CoordinateTransform<float>*     m_coordinateTransform;

	/// Argument count
	int 							m_argc;

	/// Argument values
	char**							m_argv;


};

} /* namespace lvr2 */

#endif /* INCLUDE_LIBLVR2_CONFIG_BASEOPTION_HPP_ */
