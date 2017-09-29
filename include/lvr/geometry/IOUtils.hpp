#ifndef IOUTILS_HPP
#define IOUTILS_HPP

#include <lvr/io/Timestamp.hpp>
#include <lvr/io/Model.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <boost/filesystem.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <vector>

namespace lvr
{

/**
 * @brief   Returns a Eigen 4x4 maxtrix representation of the transformation
 *          represented in the given frame file.
 */
Eigen::Matrix4d getTransformationFromFrames(boost::filesystem::path& frames);

/**
 * @brief   Returns a Eigen 4x4 maxtrix representation of the transformation
 *          represented in the given pose file.
 */
Eigen::Matrix4d getTransformationFromPose(boost::filesystem::path& pose);

/**
 * @brief   Transforms an slam6d transformation matrix into an Eigen 4x4 matrix.
 */
Eigen::Matrix4d buildTransformation(double* alignxf);

/***
 * @brief   Counts the number of points (i.e., lines) in the given file. We
 *          assume that it is an plain ASCII with one point per line.
 */
size_t countPointsInFile(boost::filesystem::path& inFile);

/**
 * @brief   Writes a Eigen transformation into a .frames file
 *
 * @param   transform   The transformation
 * @param   framesOut   The target file.
 */
void writeFrames(Eigen::Matrix4d transform, const boost::filesystem::path& framesOut);

/**
 * @brief   Writes the given model to the given file
 *
 * @param   model       A LVR model
 * @param   outfile     The target file.
 * @return  The number of points writen to the target file.
 */
size_t writeModel( ModelPtr model,const  boost::filesystem::path& outfile);

/**
 * @brief   Writes the points stored in the given model to the fiven output
 *          stream. This function is used to apend point cloud data to an
 *          already existing ASCII file..
 *
 * @param   model       A model containing point cloud data
 * @param   out         A output stream
 * @param   nocolor     If set to true, the color information in the model
 *                      is ignored.
 * @return  The number of points written to the output stream.
 */
size_t writePointsToASCII(ModelPtr model, std::ofstream& out, bool nocolor = false);

/**
 * @brief   Computes the reduction factor for a given target size (number of
 *          points) when reducing a point cloud using a modulo filter.
 *
 * @param   model       A model containing point cloud data
 * @param   targetSize  The desired number of points in the reduced model
 *
 * @return  The parameter n for the modulo filter (which means that you only
 *          have to write only every nth point to have approximately \ref
 *          targetSize points in the reduced point cloud.
 */
size_t getReductionFactor(ModelPtr model, size_t targetSize);

/**
 * @brief   Computes the reduction factor for a given target size (number of
 *          points) when reducing a point cloud loaded from an ASCII file
 *          using a modulo filter.
 *
 * @param   inFile      An ASCII file containing point cloud data
 * @param   targetSize  The desired number of points in the reduced model
 *
 * @return  The parameter n for the modulo filter (which means that you only
 *          have to write only every nth point to have approximately \ref
 *          targetSize points in the reduced point cloud.
 */
size_t getReductionFactorASCII(boost::filesystem::path& inFile, size_t targetSize);

/**
 * @brief   Transforms (scale and switch coordinates) and reduces a model
 *          containing point cloud data using a modulo filter. Use this
 *          fuction the convert between different coordinate systems.
 *
 * @param   model       A model containing point cloud data
 * @param   modulo      The reduction factor for the modulo filter. Set to
 *                      1 to keep the original resolution.
 * @param   sx          Scaling factor in x direction
 * @param   sy          Scaling factor in y direction
 * @param   sz          Scaling factor in z direction
 * @param   xPos        Position of the x position in the input data, i.e,
 *                      "which array position has the x coordinate that is written
 *                      to the output data in the input data"
 * @param   yPos        Same as with xPos for y.
 * @param   zPos        Same as with xPos for z.
 */
void transformAndReducePointCloud(ModelPtr model, int modulo, int sx, int sy, int sz, int xPos, int yPos, int zPos);

/**
 * @brief   Transforms a model containing a point cloud according to the given
 *          transformation (usually from a .frames file)
 * @param   A model containing point cloud data.
 * @param   A transformation.
 */
void transformPointCloud(ModelPtr model, Eigen::Matrix4d transformation);

/**
 * @brief   Transforms the given point buffer according to the transformation
 *          stored in \ref transformFile and appends the transformed points and
 *          normals to \ref pts and \ref nrm.
 *
 * @param   buffer          A PointBuffer
 * @param   transformFile   The input file name. The fuction will search for transformation information
 *                          (.pose or .frames)
 * @param   pts             The transformed points are added to this vector
 * @param   nrm             The transformed normals are added to this vector
 */
void transformPointCloudAndAppend(PointBufferPtr& buffer, boost::filesystem::path& transfromFile, std::vector<float>& pts, std::vector<float>& nrm);

/**
 * @brief   Writes the points and normals (float triples) stored in \ref p and \ref n
 *          to the given output file. Attention: The data is converted to a PointBuffer
 *          structure to be able to use the IO library, which results in a considerable
 *          memory overhead.
 *
 * @param   p               A vector containing point definitions.
 * @param   n               A vector containing normal information.
 * @param   outfile         The target file.
 */
void writePointsAndNormals(std::vector<float>& p, std::vector<float>& n, std::string outfile);


} // namespace lvr

#endif // IOUTILS_HPP

