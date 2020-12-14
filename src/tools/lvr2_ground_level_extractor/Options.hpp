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
 * Options.h
 *
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <float.h>

#include "lvr2/config/BaseOption.hpp"


namespace ground_level_extractor
{

/**
 * @brief A class to parse the program options for the reconstruction
 *        executable.
 */
class Options : public lvr2::BaseOption{
public:

    /**
     * @brief   Ctor. Parses the command parameters given to the main
     *          function of the program
     */
    Options(int argc, char** argv);
    virtual ~Options();

    /**
     * @brief   Returns the input file name
     */
    string  getInputFileName() const;

    /**
     * @brief   Returns the output file name
     */
    string  getOutputFileName() const;

    /**
     * @brief   Returns the specified extraction method
     */
    string  getExtractionMethod() const;

    /**
     * @brief   Returns the resolution of the model
     */
    float getResolution() const;

    /**
     * @brief   Returns the input GeoTIFF name
     */
    string getInputGeoTIFF() const;

    /**
     * @brief   Returns the input reference pairs file
     */
    string getInputReferencePairs() const;

    /**
     * @brief   Returns the number of the first band that should be extracted from the GeoTIFF
     */
    int getStartingBand() const;

    /**
     * @brief   Returns the number of bands 
     */
    int getNumberOfBands() const;

    /**
     * @brief   Returns the target coordinate system
     */
    string  getTargetSystem() const;

    /**
     * @brief   Returns the name of the color scale
     */
    string  getColorScale() const;

    /**
     * @brief   Returns the number of neighbors
     */
    int getNumberNeighbors() const;

    /**
     * @brief   Returns minimum radius size
     */
    float getMinRadius() const;

    /**
     * @brief   Returns maximum radius size
     */
    float getMaxRadius() const;

    /**
     * @brief   Returns the number of steps to reach the maximum radius
     */
    int getRadiusSteps() const;

    /**
     * @brief   Returns the size of the small window
     */
    int getSWSize() const;

    /**
     * @brief   Returns the small window threshold
     */
    float getSWThreshold() const;

    /**
     * @brief   Returns the size of the large window
     */
    int getLWSize() const;

    /**
     * @brief   Returns the large window threshold
     */
    float getLWThreshold() const;

    /**
     * @brief   Returns the slope threshold 
     */
    float getSlopeThreshold() const;

    /*
     * prints information about needed command-line-inputs e.g: input-file (ply)
     */
    bool printUsage() const;


private:

    /// Method used for DTM creation
    string                          m_method;

    ///Resolution of the mesh
    float                           m_resolution;

    ///First Band to be extracted
    int                             m_startingBand;

    ///Number of Bands to extract
    int                             m_numberBands;

    ///Color scale for textures
    string                          m_colorScale;

    ///Target coordinate system
    string                          m_target;

    ///Number of Neighbors
    int                             m_numberNeighbors;

    ///Minimum Radius for IMA
    float                           m_minRadius;

    ///Maximum Radius for IMA
    float                           m_maxRadius;

    ///Amount of steos to reach Maximum Radius in IMA
    int                             m_radiusSteps;

    ///Size of the Small Window in THM
    int                             m_swSize;

    ///Threshold of the Small Window in THM
    float                           m_swThreshold;

    ///Size of the Large Window in THM
    int                             m_lwSize;

    ///Threshold of the Large Window in THM
    float                           m_lwThreshold;

    ///Threshold of the Slope's Angle in THM
    float                           m_slopeThreshold;
    
};


/// Overlaoeded outpur operator
inline std::ostream& operator<<(std::ostream& os, const Options &o)
{
    //o.printTransformation(os);

    std::cout << "##### Input File Name: " << o.getInputFileName() << std::endl;
    std::cout << "##### Output File Name: " << o.getOutputFileName() + ".ply/.obj" << std::endl;
    std::cout << "##### Extraction Method: " << o.getExtractionMethod() << std::endl;
    std::cout << "##### Resolution: " << o.getResolution() << std::endl;
    std::cout << "##### GeoTIFF: " << o.getInputGeoTIFF()[0] << std::endl;
    std::cout << "##### Reference Points File: " << o.getInputReferencePairs() << std::endl;
    std::cout << "##### Starting Band: " << o.getStartingBand() << std::endl;
    std::cout << "##### Number of Bands: " << o.getNumberOfBands() << std::endl;
    std::cout << "##### Target System: " << o.getTargetSystem() << std::endl;
    std::cout << "##### Color Scale: " << o.getColorScale() << std::endl;
    std::cout << "##### Number of Neighbors: " << o.getNumberNeighbors() << std::endl;
    std::cout << "##### Min. Radius: " << o.getMinRadius() << std::endl;
    std::cout << "##### Max. Radius: " << o.getMaxRadius() << std::endl;
    std::cout << "##### Number of Radius Steps: " << o.getRadiusSteps() << std::endl;
    std::cout << "##### Small Window Size: " << o.getSWSize() << std::endl;
    std::cout << "##### Small Window Threshold: " << o.getSWThreshold() << std::endl;
    std::cout << "##### Large Window Size: " << o.getLWSize() << std::endl;
    std::cout << "##### Large Window Threshold: " << o.getLWThreshold() << std::endl;
    std::cout << "##### Slope Threshold: " << o.getSlopeThreshold() << std::endl;    

    return os;
}

} // namespace ground_level_extractor


#endif /* OPTIONS_H_ */
