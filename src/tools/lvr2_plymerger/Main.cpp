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

#include "Options.hpp"

#include <rply.h>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Progress.hpp"

#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <tuple>

using std::ofstream;
using std::cout;
using std::endl;
using std::pair;

using namespace lvr2;

void parsePLYHeader(
        string filename,
        size_t& numPoints,
        bool& hasNormals,
        bool& hasColors)
{
    numPoints   = 0;
    hasNormals  = false;
    hasColors   = false;

    p_ply ply = ply_open(filename.c_str(), NULL, 0, NULL);

    if(!ply)
    {
        std::cout << timestamp << "Could not open '" << filename << "." << endl;
        return;
    }

    if ( !ply_read_header( ply ) )
    {
       std::cout << timestamp << "Could not read header." << endl;
       return;
    }

    p_ply_element element = 0;
    char buf[256] = "";
    const char * name = buf;
    long int n;

    bool found_vertices = false;
    bool found_points   = false;
    bool found_faces    = false;

    while( (element = ply_get_next_element(ply, element)))
    {
         ply_get_element_info( element, &name, &n );
         if(!strcmp( name, "vertex" ))
         {
             found_vertices = true;
         }
         else if(!strcmp( name, "point" ))
         {
             found_points = true;
         }
         else if(!strcmp( name, "face" ))
         {
             found_faces = true;
         }
    }

    // Parsed header and saved meta information. Close file.
     ply_close(ply);

    // Found faces and vertices without additional point field. Assuming
    // that the .ply does not contain point cloud data
    if(found_faces && found_vertices && !found_points)
    {
        cout << timestamp << "Warning: While parsing '" << filename << "': Found mesh data without points." << endl;
        //return;
    }


    // Re-open file and try to extract properties of found point cloud data.
    ply = ply_open(filename.c_str(), NULL, 0, NULL);

    if(!ply)
    {
        std::cout << timestamp << "Could not open '" << filename << "." << endl;
        return;
    }

    if ( !ply_read_header( ply ) )
    {
       std::cout << timestamp << "Could not read header." << endl;
       return;
    }

    element = NULL;
    while ((element = ply_get_next_element( ply, element )))
    {
        ply_get_element_info(element, &name, &n);

        // Found vertices without points -> scan for vertex field
        if(found_vertices && !found_points)
        {
            if (!strcmp( name, "vertex" ) )
            {
//                cout << timestamp << "Reading points from vertex field." << endl;
//                cout << timestamp << "File contains " << n << " points." << endl;
                p_ply_property prop = NULL;
                numPoints = n;
                while ((prop = ply_get_next_property(element, prop)))
                {
                    ply_get_property_info( prop, &name, NULL, NULL, NULL );
                    if ( !strcmp( name, "red" ) ||  !strcmp( name, "r" ))
                    {
                        //cout << timestamp << "Found colors." << endl;
                        hasColors = true;
                    }
                    else if(!strcmp( name, "nx"))
                    {
                        //cout << timestamp << "Found normals." << endl;
                        hasNormals = true;
                    }
                }
            }
        }
        else if(found_points)
        {
            if (!strcmp( name, "point" ) )
            {
//                cout << timestamp << "Reading points from point field." << endl;
//                cout << timestamp << "File contains " << n << " points." << endl;
                p_ply_property prop = NULL;
                numPoints = n;
                while ((prop = ply_get_next_property(element, prop)))
                {
                    ply_get_property_info( prop, &name, NULL, NULL, NULL );
                    if ( !strcmp( name, "red" ) ||  !strcmp( name, "r" ))
                    {
                        //cout << timestamp << "Found colors." << endl;
                        hasColors = true;
                    }
                    else if(!strcmp( name, "nx"))
                    {
                        //cout << timestamp << "Found normals." << endl;
                        hasNormals = true;
                    }
                }
            }
        }
    }
    ply_close(ply);
}


void writePLYHeader(ofstream& outfile, size_t numPoints, bool writeColors, bool writeNormals)
{
    outfile << "ply" << endl;
    outfile << "format binary_little_endian 1.0" << endl;
    outfile << "element vertex " << numPoints << endl;
    outfile << "property float x" << endl;
    outfile << "property float y" << endl;
    outfile << "property float z" << endl;
    if(writeColors)
    {
        outfile << "property uchar red" << endl;
        outfile << "property uchar green" << endl;
        outfile << "property uchar blue" << endl;
    }

    if(writeNormals)
    {
        outfile << "property float nx" << endl;
        outfile << "property float ny" << endl;
        outfile << "property float nz" << endl;
    }
    outfile << "end_header" << endl;
}

void addToFile(ofstream& out, string filename)
{
    ModelPtr model = ModelFactory::readModel(filename);
    PointBufferPtr pointBuffer = model->m_pointCloud;

    size_t np, nn, nc;
    size_t w_color;
    np = pointBuffer->numPoints();
    nc = nn = np;

    floatArr points = pointBuffer->getPointArray();
    ucharArr colors = pointBuffer->getColorArray(w_color);
    floatArr normals = pointBuffer->getNormalArray();

    // Determine size of single point
    size_t buffer_size = 3 * sizeof(float);

    if(colors)
    {
        buffer_size += w_color * sizeof(unsigned char);
    }

    if(normals)
    {
        buffer_size += 3 * sizeof(float);
    }

    char* buffer = new char[buffer_size];

    for(size_t i = 0; i < np; i++)
    {
        char* ptr = &buffer[0];

        // Write coordinates to buffer
        *((float*)ptr) = points[3*i];
        ptr += sizeof(float);
        *((float*)ptr) = points[3*i+1];
        ptr += sizeof(float);
        *((float*)ptr) = points[3*i+2];

        // Write colors to buffer
        if(colors)
        {
            ptr += sizeof(float);
            *((unsigned char*)ptr) = colors[3 * i];
            ptr += sizeof(unsigned char);
            *((unsigned char*)ptr) = colors[3 * i + 1];
            ptr += sizeof(unsigned char);
            *((unsigned char*)ptr) = colors[3 * i + 2];
        }

        if(normals)
        {
            ptr += sizeof(unsigned char);
            *((float*)ptr) = normals[3*i];
            ptr += sizeof(float);
            *((float*)ptr) = normals[3*i+1];
            ptr += sizeof(float);
            *((float*)ptr) = normals[3*i+2];
        }
        out.write((const char*)buffer, buffer_size);
    }

    delete[] buffer;
}

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
    ply_merger::Options options(argc, argv);

    // Stores the names of the found ply files in the input directory.
    vector<pair<string, size_t> > ply_file_names;

    bool mergeColors = false;
    bool mergeNormals = false;
    size_t totalNumPoints = 0;

    // Check given arguments
    boost::filesystem::path inputDir(options.inputDir());
    if(boost::filesystem::exists(inputDir) && boost::filesystem::is_directory(inputDir))
    {
        // Loop over directory and store names of all .ply files
        boost::filesystem::directory_iterator end;
        for(boost::filesystem::directory_iterator it(inputDir); it != end; ++it)
        {
            std::string extension = it->path().extension().string();
            if(extension == ".ply")
            {
                ply_file_names.push_back(pair<string, size_t>(it->path().string(), 0));
            }
        }

        size_t numFilesWithNormals = 0;
        size_t numFilesWithColors = 0;

        for(auto it = ply_file_names.begin(); it != ply_file_names.end(); it++)
        {
            size_t numPoints = 0;
            bool hasColors = false;
            bool hasNormals = false;
            parsePLYHeader(it->first, numPoints, hasNormals, hasColors);

            if(hasColors)
            {
                numFilesWithColors++;
            }

            if(hasNormals)
            {
                numFilesWithNormals++;
            }

            it->second = numPoints;
            totalNumPoints += numPoints;
        }

        cout << numFilesWithNormals << " " << ply_file_names.size() << endl;

        // Check if all files have same structure
        if(numFilesWithColors > 0 && numFilesWithColors == ply_file_names.size())
        {
            mergeColors = true;
        }

        if(numFilesWithNormals > 0 && numFilesWithNormals == ply_file_names.size())
        {
            mergeNormals = true;
        }

        cout << timestamp << "Parsed directory. Reading " << totalNumPoints << " points from " << ply_file_names.size() << " files." << endl;
        if(mergeNormals)
        {
            cout << timestamp << "Merging normals." << endl;
        }

        if(mergeColors)
        {
            cout << timestamp << "Merging colors." << endl;
        }
    }
    else
    {
        std::cout << timestamp << options.inputDir() << " does not exist or is not a directory." << std::endl;
    }

    string outfile_name = options.outputFile();
    ofstream out;
    out.open(outfile_name.c_str(), std::ios::binary);
    writePLYHeader(out, totalNumPoints, mergeColors, mergeNormals);

    size_t maxChunkPoints = 1e7;
    auto it = ply_file_names.begin();

    PacmanProgressBar progress(ply_file_names.size(), "Merging...");

    while(it != ply_file_names.end())
    {
        size_t pointsInChunk = 0;
        vector<string> filesInChunk;
        do
        {
            filesInChunk.push_back(it->first);
            pointsInChunk += it->second;
            ++it;
        }
        while(it != ply_file_names.end() && pointsInChunk < maxChunkPoints);


        for(auto chunkIt: filesInChunk)
        {
            addToFile(out, chunkIt);
        }

        for(size_t c = 0; c < filesInChunk.size(); c++)
        {
            ++progress;
        }
    }

	return 0;
}

