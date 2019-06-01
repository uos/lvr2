
#ifndef LAS_VEGAS_OPTIONS_HPP
#define LAS_VEGAS_OPTIONS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

using std::ostream;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace hdf5togeotiff
{

    using namespace boost::program_options;

    /**
     * @brief A class to parse the program options for the extraction of radiometric data from a HDF5 dataset
     */
    class Options {
    public:

        /**
         * @brief 	Ctor. Parses the command parameters given to the main
         * 		  	function of the program
         */
        Options(int argc, char** argv);
        virtual ~Options();

        string  getH5File()         const { return m_variables["h5"].as<string>(); }
        string  getGTIFFFile()      const { return m_variables["gtif"].as<string>(); }
        size_t  getMinChannel()     const { return m_variables["min"].as<size_t>(); }
        size_t  getMaxChannel()     const { return m_variables["max"].as<size_t>(); }
        string  getPositionCode()   const { return m_variables["pos"].as<string>(); }

    private:
        /// The internally used variable map
        variables_map                   m_variables;

        /// The internally used option description
        options_description             m_descr;

        /// The internally used positional option description
        positional_options_description  m_pdescr;

    };

    /// Overloaded output operator
    inline ostream& operator<<(ostream& os, const Options &o)
    {
        cout << "##### Porgram options: " << endl;

        return os;
    }
}

#endif //LAS_VEGAS_OPTIONS_HPP
