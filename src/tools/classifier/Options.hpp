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
 * @file Options.h
 * @author Simon Herkenhoff <sherkenh@uni-osnabrueck.de>
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <float.h>

using std::ostream;
using std::cout;
using std::endl;
using std::string;
using std::vector;


namespace reconstruct{

	using namespace boost::program_options;

	/**
	 * @brief A class to parse the program options for the reconstruction
	 * executable.
	 */
	class Options {
		public:

			/**
			 * @brief Ctor. Parses the command parameters given to the main function of
			 * the program.
			 */
			Options(int argc, char** argv);
			virtual ~Options();

		private:

	};

}

#endif

