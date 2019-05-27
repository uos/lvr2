/**
 * @brief Main file for HDF5 to GeoTIFF or BSQ to HDF5 conversion.
 * @author ndettmer <ndettmer@uos.de>
 */

#include <lvr2/io/HDF5IO.hpp>
#include <lvr2/io/ScanDataManager.hpp>

#include <boost/range/iterator_range.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include <string>
#include <fstream>

#include <sys/stat.h>

#include "lvr2/io/GeoTIFFIO.hpp"
#include "Options.hpp"


using namespace lvr2;

void printUsage()
{
    std::cout   << "This program converts either the radiometric data from a HDF5 file into a GeoTIFF file "
                << "or a classification BSQ file into a HDF5 Dataset."
                << std::endl
                << "Usage: ./lvr2_hdf5togeotiff <input path should be .tif or .h5> "
                << "<output path should be .tif or .h5>"
                << std::endl;
}

/**
 * @brief Extraction of radiometric data from a given HDF5 file into a new GeoTIFF file in given output path
 * @param input_filename Path to the input HDF5 file formatted due to lvr_2 convention
 * @param position_code 5 character code of the scan position (e.g. 00000)
 * @param output_filename Path to the output GeoTIFF file
 * @return standard C++ return value
 */
int processConversionHDFtoGTIFF(std::string input_filename,
        std::string position_code, std::string output_filename, size_t channel_min, size_t channel_max)
{
    /*------------------- HDF5 INPUT ------------------------*/
    HDF5IO hdf5(input_filename, false);
    std::vector<size_t> dim;

    // extract radiometric data
    std::string groupname = "raw/spectral/position_" + position_code;
    std::string datasetname = "spectral";
    boost::shared_array<uint16_t> spectrals = hdf5.getArray<uint16_t>(groupname, datasetname, dim);

    // extract array dimension information
    size_t num_channels = dim[0];
    size_t num_rows = dim[1];
    size_t num_cols = dim[2];

    if(num_channels < channel_max)
    {
        std::cout << "The Dataset has only " << num_channels << " channels. Using this as upper boundary." << std::endl;
        channel_max = num_channels;
    }
    num_channels = channel_max - channel_min;

    GeoTIFFIO gtifio(output_filename, num_cols, num_rows, num_channels);

    /*--------------- FILE CONVERSION --------------------*/
    // for each channel create a cv::Mat containing the spectral intensity data for the channel ...
    for(size_t channel = 0; channel < num_channels; channel++)
    {
        cv::Mat *mat = new cv::Mat(num_rows, num_cols, CV_16UC1);
        for(size_t row = 0; row < num_rows; row++)
        {
            for(size_t col = 0; col < num_cols; col++)
            {
                mat->at<uint16_t>(row, col) = spectrals.get()[channel * num_cols * num_rows + row * num_cols + col];
            }
        }
        // ... and write it to the output GeoTIFF file
        int ret = gtifio.writeBand(mat, channel + 1);
        if (ret != 0)
        {
            return ret;
        }
    }

}

int main(int argc, char**argv)
{
    hdf5togeotiff::Options options(argc, argv);

    /*--------------- GET PROGRAM OPTIONS --------------------*/

    boost::filesystem::path input_filename(options.getH5File());
    std::string input_extension = boost::filesystem::extension(input_filename);

    boost::filesystem::path output_filename(options.getGTIFFFile());
    std::string  output_extension = boost::filesystem::extension(output_filename);
    
    size_t


    /*---------------- PREPARE CONVERSION -------------------*/

    size_t channel_min = 0;
    size_t channel_max = UINT_MAX;

    std::cout << "Please enter lower channel boundary:";
    std::cin >> channel_min;
    if (channel_min < 0)
    {
        std::cout << "You entered a negative lower boundary. Using 0." << endl;
        channel_min = 0;
    }

    std::cout << "Please enter upper channel boundary:";
    std::cin >> channel_max;
    if (!channel_max)
    {
        channel_max = UINT_MAX;
    }


    boost::filesystem::path output_dir = output_filename.parent_path();
    if (!boost::filesystem::exists(output_dir))
    {
        boost::filesystem::create_directory(output_dir);
    }

    std::string position_code;
    std::cout << "Please enter the correct scan position code of five integers (e.g. 00000):";
    std::cin >> position_code;

    if (position_code.length() != 5)
    {
        printUsage();
        return 0;
    }

    std::cout << "Starting conversion..." <<  std::endl;
    if (processConversionHDFtoGTIFF(input_filename.string(), position_code, output_filename.string(), channel_min, channel_max) < 0)
    {
        std::cout << "An Error occurred during conversion." << std::endl;
    }

    return 0;
}
