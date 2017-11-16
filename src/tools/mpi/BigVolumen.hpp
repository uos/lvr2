//
// Created by imitschke on 17.07.17.
//

#ifndef LAS_VEGAS_BigVolumen_HPP
#define LAS_VEGAS_BigVolumen_HPP


#include <string>
#include <unordered_map>
#include <utility>

#include <omp.h>

#include "lvr/io/DataStruct.hpp"
#include "lvr/geometry/BoundingBox.hpp"
#include "lvr/geometry/Vertex.hpp"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <fstream>
#include <sstream>




class BigVolumen
{





public:

    struct VolumeCellInfo
    {
        VolumeCellInfo() : size(0), overlapping_size(0) {}
        size_t size;
        size_t overlapping_size;
        size_t ix;
        size_t iy;
        size_t iz;
        std::string path;
        lvr::BoundingBox<lvr::Vertexf> bb;
        std::ofstream ofs_points;
        std::ofstream ofs_normals;
        std::ofstream ofs_colors;
    };

    /**
     * Constructor:
     * @param cloudPath path to PointCloud in ASCII xyz Format // Todo: Add other file formats
     * @param voxelsize
     */
    BigVolumen(std::vector<std::string>, float voxelsize, float overlapping_size = 0, float scale = 0);

    /**
     * @return Number of voxels
     */
    size_t size();

    /**
     * @return Number Points
     */
    size_t pointSize();


    lvr::BoundingBox<lvr::Vertexf>& getBB(){return m_bb;}

    virtual ~BigVolumen();

    inline size_t hashValue(size_t i, size_t j, size_t k)
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

    inline bool hasColors()
    {
        return m_has_color;
    }
    inline bool hasNormals()
    {
        return m_has_normal;
    }

    inline std::unordered_map<size_t, VolumeCellInfo >* getCellinfo(){return &m_gridNumPoints;};
private:

    inline int calcIndex(float f)
    {
        return f < 0 ? f-.5:f+.5;
    }

    size_t m_maxIndexSquare;
    size_t m_maxIndex;
    size_t m_maxIndexX;
    size_t m_maxIndexY;
    size_t m_maxIndexZ;
    size_t m_numPoints;
    float m_voxelSize;
    bool m_extrude;
    omp_lock_t m_lock;

    bool m_has_normal;
    bool m_has_color;

    boost::iostreams::mapped_file m_PointFile;
    boost::iostreams::mapped_file m_NomralFile;
    boost::iostreams::mapped_file m_ColorFile;
    lvr::BoundingBox<lvr::Vertexf> m_bb;
    std::unordered_map<size_t, VolumeCellInfo > m_gridNumPoints;
    float m_scale;


};

#endif //LAS_VEGAS_BigVolumen_HPP
