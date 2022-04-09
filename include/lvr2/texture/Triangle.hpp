#pragma once

// lvr2 includes
#include "lvr2/types/MatrixTypes.hpp"

// Eigen includes
#include <Eigen/Dense>

// std includes
#include <array>

/**
 * @author Justus Braun
 * @date 17.03.2022
 */
namespace lvr2
{
/**
 * @brief Represents a triangle, allows for computation of barycentric coordinates 
 * 
 */
template <typename Vec = Eigen::Vector3f, typename Scalar = float>
class Triangle
{
public:
using BarycentricCoords = Vector3<Scalar>;

private:
    // Points
    Vec m_a; // Point a
    Vec m_b; // Point b
    Vec m_c; // Point c
    // Vectors
    Vec m_AB; // Vec from a to b
    Vec m_BC; // Vec from b to c
    Vec m_CA; // Vec from c to a
    
    Scalar m_area; // Area of the triangle
    Scalar m_areaInverse; // 1 / area

    inline void init();

public:
    /**
     * @brief Construct a new Triangle object
     * 
     * @param a a point in clockwise order
     * @param b a point in clockwise order
     * @param c a point in clockwise order
     */
    Triangle(Vec a, Vec b, Vec c);

    /**
     * @brief Construct a new Triangle object
     * 
     * @param array An array containing 3 Vec objects
     */
    Triangle(const std::array<Vec, 3UL>& array);

    /**
     * @brief Returns the precalculated area of this Triangle
     * 
     * @return Scalar Area of the Triangle
     */
    Scalar area() const { return m_area; };

    /**
     * @brief Calculates the barycentric coordinates of a point with respect to this triangle.
     * Use point(BarycentricCoords) to do the opposite
     * 
     * @param point 
     * @return BarycentricCoords
     */
    BarycentricCoords barycentric(Vec point) const;

    /**
     * @brief Calculates the cartesian coordinates from barycentric coordinates
     * 
     * @param barycentric 
     * @return Vec 
     */
    Vec point(BarycentricCoords barycentric) const;

    /**
     * @brief Returns the min and max corners of the axis aligned bounding box
     * 
     * @return std::pair<Vec, Vec> first is the min corner and second the max corner
     */
    std::pair<Vec, Vec> getAABoundingBox() const;

    /**
     * @brief Calculates if a point lies within the triangle
     * 
     * @param point 
     * @return true 
     * @return false 
     */
    bool contains(Vec point) const;

    /**
     * @brief Calculates the triangles center
     * 
     * @return Vec 
     */
    Vec center() const;

    /**
     * @brief Calculates the triangles normal
     * 
     * @return Vec The normal of the triangle with length 1
     */
    Vec normal() const;
};

} // namespace lvr2

#include "Triangle.tcc"