#include "Options.hpp"

namespace hdf5togeotiff
{
    Options::Options(int argc, char **argv) : m_descr("Supported options")
    {
        // Create option descriptions

        m_descr.add_options()
                ("help", "Produce help message")
                ("h5", value<string>()->default_value(""), "Input HDF5 dataset containing hyperspectral data.")
                ("gtif", value<string>()->default_value("gtif.tif"), "Output GeoTIFF raster dataset containing hyperspectral data.")
                ("min", value<int>()->default_value(0), "Minimum hyperspectral band to be included in conversion.")
                ("max", value<int>()->default_value(150), "Maximum hyperspectral band to be included in conversion.")
                ("pos", value<string>()->default_value("00000"), "5 character identification code of scan position to be converted.");

        // Parse command line and generate variables map
        store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
        notify(m_variables);

        if(m_variables.count("help"))
        {
            ::std::cout << m_descr << ::std::endl;
            exit(-1);
        }

    }

    Options::~Options() {}
} // namespace hdf5togeotiff
