#include "Options.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/ScanIOUtils.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/hdf5/ScanProjectIO.hpp"
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

using namespace lvr2;

using BaseHDF5IO = lvr2::Hdf5IO<>;

// Extend IO with features (dependencies are automatically fetched)
using HDF5IO =
    BaseHDF5IO::AddFeatures<lvr2::hdf5features::ScanProjectIO, lvr2::hdf5features::ArrayIO>;

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

    bool exitsts = false;
    ScanProjectPtr existingScanProject;

    // check if HDF5 already exists
    if (boost::filesystem::exists(outputPath))
    {
        std::cout << timestamp << "File already exists. Expanding File..." << std::endl;

        // get existing scans
        hdf.open(outputPath.string());
        existingScanProject = hdf.loadScanProject();
        exitsts = true;
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
    if (exitsts)
    {
        for (ScanPositionPtr scanPosPtr : scanProject->positions)
        {
            existingScanProject->positions.push_back(scanPosPtr);
        }
        hdf.save(existingScanProject);
    }
    else
    {
        hdf.save(scanProject);
    }

    if (m_usePreviews)
    {
        for (int i = 0; i < scanProject->positions.size(); i++)
        {
            char buffer[128];
            sprintf(buffer, "%08d", i);
            string nr_str(buffer);
            std::string previewGroupName = "/preview/" + nr_str;

            std::cout << timestamp << "Generating preview for position " << nr_str << std::endl;

            ScanPositionPtr scanPositionPtr = scanProject->positions[i];

            ScanPtr scanPtr = scanPositionPtr->scans[0];
            floatArr points = scanPtr->points->getPointArray();

            if (points)
            {
                size_t numPreview;
                floatArr previewData = reduceData(points,
                                                  scanPtr->points->numPoints(),
                                                  3,
                                                  m_previewReductionFactor,
                                                  &numPreview);

                std::vector<size_t> previewDim = {numPreview, 3};
                hdf.save<float>(previewGroupName, "points", previewDim, previewData);
            }
        }
    }

    std::cout << timestamp << "Program finished" << std::endl;
}
