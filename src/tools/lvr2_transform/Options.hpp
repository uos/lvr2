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
