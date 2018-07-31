#ifndef POINTBUFFER2_HPP
#define POINTBUFFER2_HPP

#include <lvr2/io/DataStruct.hpp>
#include <lvr2/io/BaseBuffer.hpp>
#include <lvr2/io/AttributeManager.hpp>

#include <map>
#include <string>

#include <boost/shared_array.hpp>
#include <iostream>

namespace lvr2
{

///
/// \brief A class to handle point information with an arbitrarily
///        large number of attribute channels. Point definitions,
///        as well as normal and color buffers, are cached outside
///        the attribute maps to allow faster access.
///        The added channels should always have the some length
///        as the point array to keep the mapping
///        between geometry (channel 'points') and the associated layers like RGB
///        colors or point normals consistent.
///
class PointBuffer2 : public BaseBuffer
{
public:    
    PointBuffer2();

    /***
     * @brief Constructs a point buffer with point the given number
     *        of point.
     *
     * @param points    An array containing point data (x,y,z).
     * @param n         Number of points
     */
    PointBuffer2(floatArr points, size_t n);

    /***
     * @brief Constructs a point buffer with point and normal
     *        information. Both arrays are exspected to have the same
     *        length.
     *
     * @param points    An array containing point data (x,y,z).
     * @param normals   An array containing normal information (nx, ny, nz)
     * @param n         Number of points
     */
    PointBuffer2(floatArr points, floatArr normals, size_t n);

    /***
     * @brief Adds points to the buffer. If the buffer already
     *        contains point cloud data, the interal buffer will
     *        be freed als well as all other attribute channels.
     */
    void setPointArray(floatArr points, size_t n);

    /***
     * @brief Adds an channel containing point normal data to the
     *        buffer.
     *
     * @param   normals A float array containing the normal data.
     *                  expected to be tuples (nx, ny, nz).
     */
    void setNormalArray(floatArr normals, size_t n);

    /***
     * @brief Generates and adds a channel for point color data
     *
     * @param   colors  Am array containing point cloud data
     * @param   n       Number of colors in the buffer
     * @param   width   Number of attributes per element. Normally
     *                  3 for RGB and 4 for RGBA buffers.
     */
    void setColorArray(ucharArr colors, size_t n, unsigned width = 3);

    /// Returns the internal point array
    floatArr getPointArray();

    /// If the buffer stores normals, the
    /// call we return an empty array, i.e., the shared pointer
    /// contains a nullptr.
    floatArr getNormalArray();

    /// If the buffer stores color information, the
    /// call we return an empty array, i.e., the shared pointer
    /// contains a nullptr.
    ucharArr getColorArray();

    /// True, if buffer contains colors
    bool hasColors() const;

    /// True, if buffer has normals
    bool hasNormals() const;

    /// Returns the number of points in the buffer
    size_t numPoints() const;
private:

    /// Point channel, 'cached' to allow faster access
    FloatChannelPtr                     m_points;

    /// Normal channel, 'cached' to allow faster access
    FloatChannelPtr                     m_normals;

    /// Color channel, 'chached' to allow faster access
    UCharChannelPtr                     m_colors;

    // Number of points in buffer
    size_t              m_numPoints;



};

using PointBuffer2Ptr = std::shared_ptr<PointBuffer2>;

} // namespace lvr2

#endif // POINTBUFFER2_HPP
