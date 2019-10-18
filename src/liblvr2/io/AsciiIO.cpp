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
 * AsciiIO.cpp
 *
 *  Created on: 09.03.2011
 *      Author: Thomas Wiemann
 */

#include <fstream>
#include <string.h>
#include <algorithm>

using std::ifstream;

#include <boost/filesystem.hpp>

#include "lvr2/io/AsciiIO.hpp"
#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"

namespace lvr2
{


ModelPtr AsciiIO::read(
        string filename,
        const int &xPos, const int& yPos, const int& zPos,
        const int &rPos, const int& gPos, const int& bPos, const int &iPos)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension(selectedFile.extension().string());

    if ( extension != ".pts" && extension != ".3d" && extension != ".xyz" && extension != ".txt" )
    {
        cout << "»" << extension << "« is not a valid file extension." << endl;
        return ModelPtr();
    }
    // Count lines in file to estimate the number of present points
    size_t lines_in_file = countLines(filename);

    if ( lines_in_file < 2 )
    {
        cout << timestamp << "AsciiIO: Too few lines in file (has to be > 2)." << endl;
        return ModelPtr();
    }

    // Open file
    ifstream in;
    in.open(filename.c_str());

    // Read first two lines, ignore the first one
    char buffer[2048];
    in.getline(buffer, 2048);
    in.getline(buffer, 2048);

    // Get number of entries in test line and analize
    int num_columns  = AsciiIO::getEntriesInLine(filename);

    // Reopen file and read data
    in.close();
    in.open(filename.c_str());

    // Again skip first line
    in.getline(buffer, 2048);

    // Buffer related variables
    size_t numPoints = 0;

    floatArr points;
    ucharArr pointColors;
    floatArr pointIntensities;

    // Alloc memory for points
    numPoints = lines_in_file - 1;
    points = floatArr( new float[ numPoints * 3 ] );
    ModelPtr model(new Model);
    model->m_pointCloud = PointBufferPtr( new PointBuffer);

    // (Some) sanity checks for given paramters
    if(rPos > num_columns || gPos > num_columns || bPos > num_columns || iPos > num_columns)
    {
        cout << timestamp << "Error: At least one attribute index is larger than the number of columns" << endl;
        // Retrun empty model
        return ModelPtr();
    }

    // Check for color and intensity
    bool has_color = (rPos > -1 && gPos > -1 && bPos > -1);
    bool has_intensity = (iPos > -1);

    // Alloc buffer memory for additional attributes
    if ( has_color )
    {
        pointColors = ucharArr( new uint8_t[ numPoints * 3 ] );
    }

    if ( has_intensity )
    {
        pointIntensities = floatArr( new float[ numPoints ] );
    }

    // Read data form file
    size_t c = 0;
    while (in.good() && c < numPoints)
    {
        float x, y, z, intensity, dummy;
        unsigned int r, g, b;

        for(int i = 0; i < num_columns; i++)
        {
            if(i == xPos)
            {
                in >> x;
            }
            else if(i == yPos)
            {
                in >> y;
            }
            else if(i == zPos)
            {
                in >> z;
            }
            else if(i == rPos)
            {
                in >> r;
            }
            else if(i == gPos)
            {
                in >> g;
            }
            else if(i == bPos)
            {
                in >> b;
            }
            else if(i == iPos)
            {
                in >> intensity;
            }
            else
            {
                in >> dummy;
            }

        }

        // Read according to determined format
        if(has_color)
        {
            pointColors[ c * 3     ] = (unsigned char) r;
            pointColors[ c * 3 + 1 ] = (unsigned char) g;
            pointColors[ c * 3 + 2 ] = (unsigned char) b;

        }

        if (has_intensity)
        {
            pointIntensities[c] = intensity;

        }

        points[ c * 3     ] = x;
        points[ c * 3 + 1 ] = y;
        points[ c * 3 + 2 ] = z;
        c++;
    }

    // Sanity check
    if(c != numPoints)
    {
        cout << timestamp << "Warning: Point count / line count mismatch: "
             << numPoints << " / " << c << endl;
    }

    // Assign buffers
    size_t numColors = 0;
    size_t numIntensities = 0;

    if(has_color)
    {
        numColors = numPoints;
        model->m_pointCloud->setColorArray(pointColors, numColors);

    }

    if(has_intensity)
    {
        numIntensities = numPoints;
        model->m_pointCloud->addFloatChannel(pointIntensities, "intensities", numIntensities, 1);

    }


    model->m_pointCloud->setPointArray(points, numPoints);

    this->m_model = model;
    return model;
}


ModelPtr AsciiIO::read(string filename)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension(selectedFile.extension().string());

    if ( extension != ".pts" && extension != ".3d" && extension != ".xyz" && extension != ".txt" )
    {
        cout << "»" << extension << "« is not a valid file extension." << endl;
        return ModelPtr();
    }
    // Count lines in file to estimate the number of present points
    int lines_in_file = countLines(filename);

    if ( lines_in_file < 2 )
    {
        cout << timestamp << "AsciiIO: Too few lines in file (has to be > 2)." << endl;
        return ModelPtr();
    }
    // Open the given file. Skip the first line (as it may
    // contain meta data in some formats). Then try to guess
    // the additional data using some heuristics that apply for
    // most data formats: If 4 values per point are, given
    // the 4th value usually is a reflectence information.
    // Six entries suggest RGB information, seven entries
    // intensity and RGB.

    // Open file
    ifstream in;
    in.open(filename.c_str());

    // Read first to lines, ignore the first one
    char buffer[2048];
    in.getline(buffer, 2048);
    in.getline(buffer, 2048);

    // Get number of entries in test line and analize
    int num_attributes  = AsciiIO::getEntriesInLine(filename) - 3;
    bool has_color      = (num_attributes == 3) || (num_attributes == 4);
    bool has_intensity  = (num_attributes == 1) || (num_attributes == 4);

    if(has_color || has_intensity)
    {
        cout << timestamp << "Autodetected the following attributes" << endl;
        cout << timestamp << "Color:     " << has_color << endl;
        cout << timestamp << "Intensity: " << has_intensity << endl;

        if(has_color && has_intensity)
        {
            return read(filename, 0, 1, 2, 4, 5, 6, 3);
        }
        else if(has_color)
        {
            return read(filename, 0, 1, 2, 3, 4, 5);
        }
        else if(has_intensity)
        {
            if(num_attributes == 1)
            {
                return read(filename, 0, 1, 2, -1, -1, -1, 3);
            }
            else
            {
                return read(filename, 0, 1, 2, -1, -1, -1, 6);
            }
        }
    }
    return read(filename, 0, 1, 2);

}



void AsciiIO::save( std::string filename )
{

    if ( !this->m_model->m_pointCloud ) {
        std::cerr << "No point buffer available for output." << std::endl;
        return;
    }

    size_t   pointcount( 0 ), buf ( 0 );

    floatArr   points;
    ucharArr   pointColors;
    floatArr   pointIntensities;

    unsigned w;
    pointcount = this->m_model->m_pointCloud->numPoints();
    points = this->m_model->m_pointCloud->getPointArray();
//    points = this->m_model->m_pointCloud->getIndexedPointArray( pointcount );


//    pointColors = this->m_model->m_pointCloud->getIndexedPointColorArray( buf );
//    pointColors = this->m_model->m_pointCloud->getColorArray(buf);
    auto colors = this->m_model->m_pointCloud->getChannel<unsigned char>("colors");
    if(colors)
    {
      pointColors = (*colors).dataPtr();
      buf = (*colors).numElements();
      /* We need the same amount of color information and points. */
      if ( pointcount != buf )
      {
        pointColors.reset();
        std::cerr << "Amount of points and color information is"
          " not equal. Color information won't be written" << std::endl;
      }
    }
    //   pointIntensities = this->m_model->m_pointCloud->getPointIntensityArray( buf );
    auto intensity = this->m_model->m_pointCloud->getChannel<float>("intensities");
    if(intensity)
    {
      pointIntensities = (*intensity).dataPtr();
      buf = (*intensity).numElements();
    

      /* We need the same amount of intensity values and points. */
      if ( pointcount != buf )
      {
        pointIntensities.reset();
        std::cerr << "Amount of points and intensity values are"
          " not equal. Intensity information will not be written." << std::endl;
      }
    }

    /* Prepare file for writing. */
    std::ofstream out( filename.c_str() );

    if ( !out.is_open() ) {
        std::cerr << "Could not open file »" << filename
            << "« for output." << std::endl;
        return;
    }

    for ( size_t i(0); i < pointcount; i++ )
    {
        out << points[i * 3] << " "
            << points[i * 3 + 1] << " "
            << points[i * 3 + 2];
        if ( pointIntensities )
        {
            out << " " << pointIntensities[i];
        }
        if ( pointColors )
        {
            /* Bad behaviour of C++ output streams: We have to cast the uchars
             * to unsigned integers. */
            out << " " << (unsigned int) pointColors[i * 3]
                << " " << (unsigned int) pointColors[i * 3 + 1]
                << " " << (unsigned int) pointColors[i * 3 + 2];
        }
        out << std::endl;
    }

    out.close();

}


size_t AsciiIO::countLines(string filename)
{
    // Open file for reading
    ifstream in(filename.c_str());

    // Count lines in file
    size_t c = 0;
    char line[2048];
    while(in.good())
    {
        in.getline(line, 1024);
        c++;
    }
    in.close();
    return c;
}


int AsciiIO::getEntriesInLine(string filename)
{

    ifstream in(filename.c_str());

    // Get first line from file and skip it (possibly metadata)
    char line[1024];
    in.getline(line, 1024);

    // Get second line -> hopefully point data
    in.getline(line, 1024);

    in.close();

    // Get number of blanks
    int c = 0;
    char* pch = strtok(line, " ");
    while(pch){
        c++;
        pch = strtok(NULL, " ");
    }

    return c;
}


} // namespace lvr
