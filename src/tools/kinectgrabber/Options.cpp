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


 /*
 * Options.cpp
 *
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#include "Options.hpp"

namespace kingrab{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		        ("help", "Produce help message")
		        ("t", value<int>(&m_t)->default_value(0), "Timeout before grabbing starts (s)")
		        ("d", value<int>(&m_d)->default_value(20), "Number of dumps.")
		        ("w", value<int>(&m_w)->default_value(100), "Wait time between dumps (ms)")
	        ;



	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

  if(m_variables.count("help")) {
    ::std::cout<< m_descr << ::std::endl;
  }

}




Options::~Options() {
	// TODO Auto-generated destructor stub
}

} // namespace reconstruct
