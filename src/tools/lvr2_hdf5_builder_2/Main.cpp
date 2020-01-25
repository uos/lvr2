#include "Options.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/io/GHDF5IO.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/ScanIOUtils.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/hdf5/ScanProjectIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;

using BaseHDF5IO = lvr2::Hdf5IO<>;

// Extend IO with features (dependencies are automatically fetched)
using HDF5IO = BaseHDF5IO::AddFeatures<lvr2::hdf5features::ScanProjectIO>;

bool m_usePreviews;
int m_previewReductionFactor;

template <typename T>
boost::shared_array<T> reduceData(boost::shared_array<T> data,
                                  size_t dataCount,
                                  size_t dataWidth,
                                  unsigned int reductionFactor,
                                  size_t* reducedDataCount)
{
    *reducedDataCount = dataCount / reductionFactor + 1;
    boost::shared_array<T> reducedData =
        boost::shared_array<T>(new T[(*reducedDataCount) * dataWidth]);

    size_t reducedDataIdx = 0;
    for (size_t i = 0; i < dataCount; i++)
    {
        if (i % reductionFactor == 0)
        {
            std::copy(data.get() + i * dataWidth,
                      data.get() + (i + 1) * dataWidth,
                      reducedData.get() + reducedDataIdx * dataWidth);

            reducedDataIdx++;
        }
    }

    return reducedData;
}

int main(int argc, char** argv)
{
    hdf5tool2::Options options(argc, argv);
    boost::filesystem::path inputDir(options.getInputDir());
    boost::filesystem::path outputDir(options.getOutputDir());

    boost::filesystem::path outputPath(outputDir / options.getOutputFile());

    m_usePreviews = options.getPreview();
    m_previewReductionFactor = options.getPreviewReductionRatio();

    HDF5IO hdf;
    uint pos = 0;

    // check if input directory exists
    if (!boost::filesystem::exists(inputDir))
    {
        std::cout << timestamp << "Error: Directory " << options.getInputDir() << " does not exist"
                  << std::endl;
        exit(-1);
    }

    // check if output directory exists
    if (!boost::filesystem::exists(outputDir))
    {
        std::cout << timestamp << "Creating directory " << options.getOutputDir() << std::endl;
        if (!boost::filesystem::create_directory(outputDir))
        {
            std::cout << timestamp << "Error: Unable to create " << options.getOutputDir()
                      << std::endl;
            exit(-1);
        }
    }

    // check if HDF5 already exists
    if (boost::filesystem::exists(outputPath))
    {
        std::cout << timestamp << "File already exists. Expanding File..." << std::endl;

        // get existing scans
        hdf.open(outputPath.string());
        HighFive::Group hfscans = hdf5util::getGroup(hdf.m_hdf5_file, "raw/scans");
        pos = hfscans.listObjectNames().size();
        std::cout << timestamp << "Using counter-increment " << pos << std::endl;
    }
    else
    {
        hdf.open(outputPath.string());
    }

    ScanProjectPtr scanProject(new ScanProject());

    // reading scan project from given directory into ScanProject
    std::cout << timestamp << "Reading ScanProject from directory" << std::endl;
    loadScanProject(inputDir, *scanProject);

    // saving ScanProject into HDF5 file
    std::cout << timestamp << "Writing ScanProject to HDF5" << std::endl;
    hdf.save(scanProject);

    std::cout << timestamp << "Program finished" << std::endl;
}
