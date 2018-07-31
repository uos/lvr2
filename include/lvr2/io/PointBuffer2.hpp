#ifndef POINTBUFFER2_HPP
#define POINTBUFFER2_HPP

#include <lvr2/io/DataStruct.hpp>
#include <lvr2/io/AttributeManager.hpp>

#include <map>
#include <string>

#include <boost/shared_array.hpp>
#include <iostream>

namespace lvr2
{

class PointBuffer2
{
public:    
    PointBuffer2();
    PointBuffer2(floatArr points, size_t n);
    PointBuffer2(floatArr points, floatArr normals, size_t n);

    void setPointArray(floatArr points, size_t n);
    void setNormalArray(floatArr normals, size_t n);
    void setColorArray(ucharArr colors, size_t n);

    void addFloatChannel(floatArr data, std::string name, size_t n, unsigned w);
    void addUCharChannel(ucharArr data, std::string name, size_t n, unsigned w);

    floatArr getPointArray();
    floatArr getNormalArray();
    floatArr getFloatArray(const std::string& name, unsigned& w);
    ucharArr getColorArray();
    ucharArr getUcharArray(const std::string& name, unsigned& w);

    FloatChannel getFloatChannel(const std::string& name);
    UCharChannel getUcharChannel(const std::string& name);

    bool hasColors() const;
    bool hasNormals() const;

    size_t numPoints() const;
private:

    // Point channel, 'cached' to allow faster access
    AttributeManager                    m_channels;
    FloatChannelPtr                     m_points;
    FloatChannelPtr                     m_normals;
    UCharChannelPtr                     m_colors;

    // Number of points in buffer
    size_t              m_numPoints;



};

using PointBuffer2Ptr = std::shared_ptr<PointBuffer2>;

} // namespace lvr2

#endif // POINTBUFFER2_HPP
