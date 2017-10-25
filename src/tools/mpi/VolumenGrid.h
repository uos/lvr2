//
// Created by imitschke on 18.10.17.
//

#ifndef LAS_VEGAS_VOLUMENGRID_H
#define LAS_VEGAS_VOLUMENGRID_H
#include <string>
#include <lvr/geometry/BoundingBox.hpp>
#include <lvr/geometry/Vertex.hpp>
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
using namespace lvr;
class VolumenGrid {
    struct CellInfo
    {
        CellInfo() : size(0), offset(0), inserted(0), dist_offset(0) {}
        size_t size;
        size_t offset;
        size_t inserted;
        size_t dist_offset;
        size_t ix;
        size_t iy;
        size_t iz;
        BoundingBox<Vertexf> bb;
    };
public:
    VolumenGrid(std::string filePath, float volumen);
    size_t getCellCount(){return grids.size();}
    BoundingBox<Vertexf> getBB(size_t i ){return m_gridNumPoints[grids[i]].bb;}
    BoundingBox<Vertexf> getBB(){return m_bb;}
    lvr::floatArr points(size_t i, size_t& numPoints);
    size_t size(){return m_gridNumPoints.size();}

private:
    BoundingBox<Vertexf> m_bb;
    size_t m_maxIndexX;
    size_t m_maxIndexY;
    size_t m_maxIndexZ;
    size_t m_maxIndex;
    size_t m_maxIndexSquare;
    bool m_has_normal;
    bool m_has_color;
    size_t m_numPoints;

    boost::iostreams::mapped_file m_PointFile;
    boost::iostreams::mapped_file m_NomralFile;
    boost::iostreams::mapped_file m_ColorFile;
    std::unordered_map<size_t, CellInfo > m_gridNumPoints;
    std::vector<size_t> grids;
    float m_volumen;
    float m_scale;
    inline int calcIndex(float f)
    {
        return f < 0 ? f-.5:f+.5;
    }
    inline size_t hashValue(size_t i, size_t j, size_t k)
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

    bool m_extrude;
};


#endif //LAS_VEGAS_VOLUMENGRID_H
