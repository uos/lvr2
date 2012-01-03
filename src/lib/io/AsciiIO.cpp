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

#include "AsciiIO.hpp"
#include "Progress.hpp"
#include "Timestamp.hpp"

namespace lssr
{

ModelPtr AsciiIO::read(string filename)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension(selectedFile.extension().c_str());

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
    bool has_color      = (num_attributes == 3) || (num_attributes == 4) || (num_attributes == 5);
    bool has_intensity  = (num_attributes == 1) || (num_attributes == 4);
    bool has_accuracy   = num_attributes == 5;
    bool has_validcolor = num_attributes == 5;

    if ( has_color ) {
        cout << timestamp << "Reading color information." << endl;
    }

    if ( has_intensity ) {
        cout << timestamp << "Reading intensity information." << endl;
    }

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
    floatArr pointConfidences;

    // Alloc memory for points
    numPoints = lines_in_file - 1;
    points = floatArr( new float[ numPoints * 3 ] );

    // Alloc buffer memory for additional attributes
    if ( has_color )
    {
        pointColors = ucharArr( new uint8_t[ numPoints * 3 ] );
    }

    if ( has_intensity )
    {
        pointIntensities = floatArr( new float[ numPoints ] );
    }

    if ( has_accuracy )
    {
        pointConfidences = floatArr( new float[ numPoints ] );
    }

    // Read data form file
    size_t c = 0;
    while (in.good() && c < numPoints)
    {

        //cout << has_intensity << " " << has_color << endl;
        //cout << c << " " << m_colors << " " << m_numPoints << endl;
        float x, y, z, i, dummy, confidence;
        unsigned int r, g, b;

        // Read according to determined format
        if(has_intensity && has_color)
        {
            in >> x >> y >> z >> i >> r >> g >> b;
            pointIntensities[c] = i;
            pointColors[ c * 3     ] = (uchar) r;
            pointColors[ c * 3 + 1 ] = (uchar) g;
            pointColors[ c * 3 + 2 ] = (uchar) b;

        }
        else if ( has_color && has_accuracy && has_validcolor )
        {
            in >> x >> y >> z >> confidence >> dummy >> r >> g >> b;
            pointConfidences[c]      = confidence;
            pointColors[ c * 3     ] = (uchar) r;
            pointColors[ c * 3 + 1 ] = (uchar) g;
            pointColors[ c * 3 + 2 ] = (uchar) b;

        }
        else if (has_intensity)
        {
            in >> x >> y >> z >> i;
            pointIntensities[c] = i;

        }
        else if(has_color)
        {
            in >> x >> y >> z >> r >> g >> b;
            pointColors[ c * 3     ] = (uchar) r;
            pointColors[ c * 3 + 1 ] = (uchar) g;
            pointColors[ c * 3 + 2 ] = (uchar) b;
        }
        else
        {
            in >> x >> y >> z;
            for(int n_dummys = 0; n_dummys < num_attributes; n_dummys++) in >> dummy;
        }
        points[ c * 3     ] = x;
        points[ c * 3 + 1 ] = y;
        points[ c * 3 + 2 ] = z;
        c++;
    }

    // Assign buffers
    ModelPtr model( new Model( PointBufferPtr( new PointBuffer)));
    model->m_pointCloud->setPointArray(           points,           numPoints );
    model->m_pointCloud->setPointColorArray(      pointColors,      numPoints );
    model->m_pointCloud->setPointIntensityArray(  pointIntensities, numPoints );
    model->m_pointCloud->setPointConfidenceArray( pointConfidences, numPoints );

    return model;
}


void AsciiIO::save( std::string filename,
        std::multimap< std::string, std::string > options, ModelPtr m )
{

    if ( m ) 
    {
        m_model = m;
    }

    /* Set PLY mode. */
    it = options.find( "comment" );
    if ( it != options.end() )
    {
        save( filename, it->second; );
    }
    else
    {
        save( filename, "" );
    }
}


void AsciiIO::save( std::string filename )
{
    save( filename, "" );
}


void AsciiIO::save( std::string filename, std::string comment )
{

    if ( !m_model->m_pointCloud ) {
        std::cerr << "No point buffer available for output." << std::endl;
        return;
    }

    size_t   pointcount( 0 ), buf ( 0 );

    coord3fArr points;
    color3bArr pointColors;
    floatArr   pointIntensities;

    points = m_model->m_pointCloud->getIndexedPointArray( pointcount );

    pointColors = m_model->m_pointCloud->getIndexedPointColorArray( buf );
    /* We need the same amount of color information and points. */
    if ( pointcount != buf )
    {
        pointColors.reset();
        std::cerr << "Amount of points and color information is"
            " not equal. Color information won't be written.\n";
    }

    pointIntensities = m_model->m_pointCloud->getPointIntensityArray( buf );
    /* We need the same amount of intensity values and points. */
    if ( pointcount != buf )
    {
        pointIntensities.reset();
        std::cerr << "Amount of points and intensity values are"
            " not equal. Intensity information will not be written.\n";
    }


    /* Prepare file for writing. */
    std::ofstream out( filename.c_str() );

    if ( !out.is_open() ) {
        std::cerr << "Could not open file »" << filename
            << "« for output." << std::endl;
        return;
    }

    /* Write comment. */
    out << "# " << comment << std::endl;

    for ( size_t i(0); i < pointcount; i++ )
    {
        out << points[i].x << " " 
            << points[i].y << " " 
            << points[i].z;
        if ( pointIntensities )
        {
            out << " " << pointIntensities[i];
        }
        if ( pointColors )
        {
            /* Bad behaviour of C++ output streams: We have to cast the uchars
             * to unsigned integers. */
            out << " " << (unsigned int) pointColors[i].r 
                << " " << (unsigned int) pointColors[i].g 
                << " " << (unsigned int) pointColors[i].b;
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
    char first_line[1024];
    in.getline(first_line, 1024);

    // Get second line -> hopefully point data
    char second_line[1024];
    in.getline(second_line, 1024);

    in.close();

    // Get number of blanks
    int c = 0;
    char* pch = strtok(second_line, " ");
    while(pch != NULL){
        c++;
        pch = strtok(NULL, " ");
    }

    in.close();

    return c;
}


} // namespace lssr
