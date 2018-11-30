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

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ostream;


namespace image_normals
{

using namespace boost::program_options;

/**
 * @brief A class to parse the program options for the reconstruction
 * executable.
 */
class Options
{
public:

	/**
	 * @brief Ctor. Parses the command parameters given to the main
	 *     function of the program
	 */
	Options(int argc, char** argv);
	virtual ~Options();


    string  inputFile() const
	{
        return (m_variables["inputFile"].as< vector<string> >())[0];
	}


    string  imageFile() const
	{
        return m_variables["img"].as<string>();
	}

    int     minH() const
	{
        return m_variables["minH"].as<int>();
	}

    int     maxH() const
	{
        return m_variables["maxH"].as<int>();
	}


    int     minV() const
	{
        return m_variables["minV"].as<int>();
	}

    int     maxV() const
	{
        return m_variables["maxV"].as<int>();
	}

    int     imageWidth() const
    {
        return m_variables["imageWidth"].as<int>();
    }


    int     imageHeight() const
    {
        return m_variables["imageHeight"].as<int>();
    }

    int     regionWidth() const
    {
        return m_variables["regionWidth"].as<int>();
    }

    int     regionHeight() const
    {
        return m_variables["regionHeight"].as<int>();
    }

    float    maxZ() const
    {
         return m_variables["maxZ"].as<float>();
    }

    float    minZ() const
    {
         return m_variables["minZ"].as<float>();
    }

    float    maxZimg() const
    {
         return m_variables["maxZimg"].as<float>();
    }

    float    minZimg() const
    {
         return m_variables["minZimg"].as<float>();
    }

    bool    optimize() const
    {
        return m_variables.count("optimize");
    }

    string    coordinateSystem() const
    {
        return m_variables["system"].as<string>();
    }

private:

	/// The internally used variable map
	variables_map m_variables;

	/// The internally used option description
	options_description m_descr;

	/// The internally used positional option desription
	positional_options_description m_pdescr;

    int         m_minH;
    int         m_maxH;
    int         m_minV;
    int         m_maxV;
    int         m_width;
    int         m_height;
    int         m_windowWidth;
    int         m_windowHeight;
    float       m_minZ;
    float       m_maxZ;
    float       m_minZimg;
    float       m_maxZimg;
    string      m_imageOut;
    string      m_system;
};

inline ostream& operator<<(ostream& os, const Options& o)
{
    os << "##### Panorama normal estimation settings #####" << endl;
    os << "Horizontal field of view\t: "   << o.minH() << " x " << o.maxH() << endl;
    os << "Vertical field of view\t\t: "   << o.minV() << " x " << o.maxV() << endl;
    os << "Image Dimensions\t\t: " << o.imageWidth() << " x " << o.imageHeight() << endl;
    os << "Z range (scan) \t\t\t: " << o.minZ() << " to " << o.maxZ() << endl;
    os << "Z range (img)\t\t\t: " << o.minZimg() << " to " << o.maxZimg() << endl;
    os << "Optimize aspect\t\t\t: " << o.optimize() << endl;
    os << "Coordinate system\t\t: " << o.coordinateSystem() << endl;
    return os;
}

} // namespace normals

#endif

