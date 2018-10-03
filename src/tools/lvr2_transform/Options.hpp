/* Copyright (C) 2012 Uni Osnabrück
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
 * options.hpp
 *
 * @date 2012-08-22
 * @author Christian Wansart <cwansart@uos.de>
 * @author Thomas Wiemann
 */

#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

using std::ostream;
using std::cout;
using std::endl;
using std::string;
using std::vector;


namespace transform 
{

  using namespace boost::program_options;

  /**
   * @brief A class to parse the program options for the reconstruction
   * 		  executable.
   */
  class Options {
    public:

      /**
       * @brief 	Ctor. Parses the command parameters given to the main
       * 		  	function of the program
       */
      Options(int argc, char** argv);
      virtual ~Options();

      /**
       * @brief	Prints a usage message to stdout.
       */
      bool	printUsage() const;

      /**
       * @brief	Returns the output file name
       */
      string 	getInputFile() const;

      /**
       * @brief	Returns the output file name
       */
      string 	getOutputFile() const;

      /**
       * @brief Returns the transform file name
       */
      string  getTransformFile() const;

      /**
       * @brief Returns true if transform file is given
       */
      bool  anyTransformFile() const;

      /**
       * @brief Returns the x axis scale
       */
      float getScaleX() const;

      /**
       * @brief Returns the y axis scale
       */
      float getScaleY() const;

      /**
       * @brief Returns the z axis scale
       */
      float getScaleZ() const;

      /**
       * @brief Returns the x axis rotation
       */
      float getRotationX() const;

      /**
       * @brief Returns the y axis rotation
       */
      float getRotationY() const;

      /**
       * @brief Returns the z axis rotation
       */
      float getRotationZ() const;

      /**
       * @brief Returns the x axis translation
       */
      float getTranslationX() const;

      /**
       * @brief Returns the y axis translation
       */
      float getTranslationY() const;

      /**
       * @brief Returns the z axis translation
       */
      float getTranslationZ() const;

      /**
       * @brief Returns true if there is any scale value
       */
      bool anyScale() const;

      /**
       * @brief Returns true if there is any x-scale value
       */
      bool anyScaleX() const;

      /**
       * @brief Returns true if there is any y-scale value
       */
      bool anyScaleY() const;

      /**
       * @brief Returns true if there is any z-scale value
       */
      bool anyScaleZ() const;

      /**
       * @brief Returns true if there is any rotation value
       */
      bool anyRotation() const;

      /**
       * @brief Returns true if there is any x-rotation value
       */
      bool anyRotationX() const;

      /**
       * @brief Returns true if there is any y-rotation value
       */
      bool anyRotationY() const;

      /**
       * @brief Returns true if there is any z-rotation value
       */
      bool anyRotationZ() const;

      /**
       * @brief Returns true if there is any translation value
       */
      bool anyTranslation() const;

      /**
       * @brief Returns true if there is any x translation value
       */
      bool anyTranslationX() const;

      /**
       * @brief Returns true if there is any y translation value
       */
      bool anyTranslationY() const;

      /**
       * @brief Returns true if there is any z translation value
       */
      bool anyTranslationZ() const;

    private:

      /// The internally used variable map
      variables_map			        m_variables;

      /// The internally used option description
      options_description 		    m_descr;

      /// The internally used positional option desription
      positional_options_description 	m_pdescr;
  };


  /// Overlaoeded outpur operator
  inline ostream& operator<<(ostream& os, const Options &o)
  {
    // TODO: Add program options?
    cout << "##### Program options: " << endl;

    return os;
  }

} // namespace applymatrix


#endif /* OPTIONS_HPP */
