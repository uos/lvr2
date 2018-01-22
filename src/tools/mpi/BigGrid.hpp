//
// Created by imitschke on 17.07.17.
//

#ifndef LAS_VEGAS_BIGGRID_HPP
#define LAS_VEGAS_BIGGRID_HPP


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
};

class BigGrid
{
public:
    /**
     * Constructor:
     * @param cloudPath path to PointCloud in ASCII xyz Format // Todo: Add other file formats
     * @param voxelsize
     */
    BigGrid(std::vector<std::string>, float voxelsize, float scale = 0, size_t bufferSize = 1024);

    BigGrid(std::string path);
    /**
     * @return Number of voxels
     */
    size_t size();

    /**
     * @return Number Points
     */
    size_t pointSize();

    /**
     * Amount of Ponts in Voxel at position i,j,k
     * @param i
     * @param j
     * @param k
     * @return amount of points, 0 if voxel does not exsist
     */
    size_t pointSize(int i, int j, int k);

    /**
     * Points of  Voxel at position i,j,k
     * @param i
     * @param j
     * @param k
     * @param numPoints, amount of points in lvr::floatArr
     * @return lvr::floatArr, containing points
     */
    lvr::floatArr points(int i, int j, int k, size_t& numPoints);

    /**
     *  Points that are within bounding box defined by a min and max point
     * @param minx
     * @param miny
     * @param minz
     * @param maxx
     * @param maxy
     * @param maxz
     * @param numPoints
     * @return lvr::floatArr, containing points
     */
    lvr::floatArr points(float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints);

    lvr::floatArr normals(float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints);

    lvr::ucharArr colors(float minx, float miny, float minz, float maxx, float maxy, float maxz, size_t& numPoints);

    size_t getSizeofBox(float minx, float miny, float minz, float maxx, float maxy, float maxz);

    void serialize(std::string path = "serinfo.ls");

    lvr::floatArr getPointCloud(size_t & numPoints);

    lvr::BoundingBox<lvr::Vertexf>& getBB(){return m_bb;}

    virtual ~BigGrid();

    inline size_t hashValue(size_t i, size_t j, size_t k)
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

    inline size_t getDistanceFileOffset(size_t hash)
    {
        if(exists(hash))
        {
            return m_gridNumPoints[hash].dist_offset;
        }
        else return 0;
    }
    inline bool exists(size_t hash)
    {
        auto it = m_gridNumPoints.find(hash);
        return it!=m_gridNumPoints.end();
    }

    inline bool hasColors()
    {
        return m_has_color;
    }
    inline bool hasNormals()
    {
        return m_has_normal;
    }
private:

    inline int calcIndex(float f)
    {
        return f < 0 ? f-.5:f+.5;
    }

    bool exists(int i, int j, int k);
    void insert(float x, float y, float z);

    size_t m_maxIndexSquare;
    size_t m_maxIndex;
    size_t m_maxIndexX;
    size_t m_maxIndexY;
    size_t m_maxIndexZ;
    size_t m_numPoints;

    size_t m_pointBufferSize;

    float m_voxelSize;
    bool m_extrude;
    omp_lock_t m_lock;

    bool m_has_normal;
    bool m_has_color;

    boost::iostreams::mapped_file m_PointFile;
    boost::iostreams::mapped_file m_NomralFile;
    boost::iostreams::mapped_file m_ColorFile;
    lvr::BoundingBox<lvr::Vertexf> m_bb;
    std::unordered_map<size_t, CellInfo > m_gridNumPoints;
    float m_scale;


};

#endif //LAS_VEGAS_BIGGRID_HPP
