/*
 * BaseOption.hpp
 *
 *  Created on: Feb 4, 2015
 *      Author: twiemann
 */

#ifndef INCLUDE_LIBLVR_CONFIG_BASEOPTION_HPP_
#define INCLUDE_LIBLVR_CONFIG_BASEOPTION_HPP_

#include <boost/program_options.hpp>
#include <iostream>

#include "io/ModelFactory.hpp"

namespace lvr
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
	CoordinateTransform				m_coordinateTransform;

	/// Argument count
	int 							m_argc;

	/// Argument values
	char**							m_argv;


};

} /* namespace lvr */

#endif /* INCLUDE_LIBLVR_CONFIG_BASEOPTION_HPP_ */
