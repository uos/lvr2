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
 * PPMIO.cpp
 *
 *  Created on:  08.09.2011
 *      Author:  Kim Rinnewitz  ( krinnewitz@uos.de )
 *  Modified on: 11.12.2011
 *      Author:  Thomas Wiemann ( twiemann@uos.de )
 *  Modified on: 15.02.2011
 *      Author:  Denis Meyer    ( denmeyer@uos.de )
 */

#include "PPMIO.hpp"
#include <iostream>
#include <fstream>
#include <string.h>

using namespace std;

namespace lssr
{

PPMIO::PPMIO()
{
    m_pixels    = 0;
    m_width   = 0;
    m_height   = 0;
}

PPMIO::PPMIO( string filename ) : m_width(0), m_height(0), m_pixels(0)
{
    // Try to open file
    ifstream in(filename.c_str());

    // Parse file
    if(in.good())
    {
        // Line buffer
        char buffer[1024];

        // Read file tag
        readLine(in, buffer);

        // Check tag
        string tag(buffer);
        if(tag == "P3")
        {
            // Read width, height and color information
            stringstream ss;
            readLine(in, buffer);
            ss << buffer << " ";
            readLine(in, buffer);
            ss << buffer << " ";

            // Read formatted data
            ss >> m_width >> m_height;

            // Alloc data
            m_pixels = new unsigned char[m_width * m_height * 3];

            // Read pixels
            int p;
            for(int i = 0; i < m_width * m_height * 3; i++)
            {
                in >> p;
                m_pixels[i] = (unsigned char)p;
            }
        }
        else
        {
            in.close();
            in.open(filename.c_str(), ios::binary);
            //	    readLine(in, buffer);
            //	    char tmp[3];
            //	    sscanf(buffer, "%s %d %d 255", tmp, &m_width, &m_height);

            string tag;
            in >> tag;


            if(tag == "P6") // TODO: hacked in for our output
            {
                int n_colors;
                in >> m_width >> m_height >> n_colors;
		in.getline(0,0);
                m_pixels = new unsigned char[m_width * m_height * 3];
                in.read((char *)m_pixels, m_width * m_height * 3);
            }
            else
            {
                cerr << "Unsupported tag, only P3 or P6 possible." << endl;
            }
        }
    }
    else
    {
        cout << "ReadPPM: Unable to open file " << filename << "." << endl;
    }
}

void PPMIO::write( string filename )
{
    ofstream out(filename.c_str());

    if(out.good())
    {
    	out<<"P6"<<" "<<m_width<<" "<<m_height<<" "<<"255"<<endl;
	out.write((char*) m_pixels, m_width * m_height * 3);
    }

    out.close();

}

void PPMIO::setDataArray( unsigned char* array, int width, int height )
{
    m_pixels = array;
    m_width = width;
    m_height = height;
}

void PPMIO::readLine( ifstream & in, char* buffer )
{
    // Read lines until no comment line was found
    do
    {
      in.getline(buffer, 256);
    }
    while(buffer[0] == '#' && in.good() );
}

}
