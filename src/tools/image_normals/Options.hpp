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

