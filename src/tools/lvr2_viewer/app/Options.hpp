#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <boost/program_options.hpp>
#include <string>
#include <vector>
#include "lvr2/config/BaseOption.hpp"

namespace viewer
{

using boost::program_options::options_description;
using boost::program_options::positional_options_description;
using boost::program_options::variables_map;
using std::string;

class Options : public lvr2::BaseOption
{
  public:
    /**
     * @brief   Ctor. Parses the command parameters given to the main
     *          function of the program
     */
    Options(int argc, char** argv);
    virtual ~Options();

    /**
     * @brief   Prints a usage message to stdout.
     */
    bool printUsage() const;

    /**
     * @brief	Returns the input file
     */
    //std::vector<std::string> getInputFiles() const;

    /**
     * @brief   Returns the input file name
     */
    std::string  getInputFileName() const;

    /**
     * @brief Returns the layers used for LOD
     *
     * @return 
     */
    std::vector<std::string> getLayers() const;

    int getCacheSize() const;

    /**
     * @brief 
     *
     * @return 
     */
    float getHighResDistance() const;

    bool isChunkedMesh() const;

};

}

#endif /* OPTIONS_HPP_ */
