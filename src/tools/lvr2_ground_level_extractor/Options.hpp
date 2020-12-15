 /*
 * Options.hpp
 *
 *  Created on: 18.11.2020
 *      Author: Mario Dyczka (mdyczka@uos.de)
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
    std::cout << "##### GeoTIFF: " << o.getInputGeoTIFF() << std::endl;
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
