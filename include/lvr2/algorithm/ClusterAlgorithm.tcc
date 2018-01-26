/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/*
 * ClusterAlgorithm.tcc
 *
 *  @date 20.07.2017
 *  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
 *  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
 *  @author Rasmus Diederichsen <rdiederichse@uni-osnabrueck.de>
 */

#include <lvr2/geometry/HalfEdgeMesh.hpp>

// #include <boost/math/constants/constants.hpp>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <lvr2/geometry/BoundingBox.hpp>

namespace lvr2
{

template<typename BaseVecT>
std::vector<VertexHandle> calculateClusterContourVertices(
    ClusterHandle clusterH,
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusterBiMap
)
{
    std::unordered_set<VertexHandle> contourVertices;

    auto cluster = clusterBiMap.getCluster(clusterH);

    // iterate over all faces in cluster
    for (auto faceH : cluster.handles)
    {
        // get edges of each face
        const auto edgesOfFace = mesh.getEdgesOfFace(faceH);


        // check for each edge if it is a contour edge
        for (auto edgeH : edgesOfFace)
        {
            const auto faces = mesh.getFacesOfEdge(edgeH);
            int numFaces = 0;

            // count how many faces the edge has that are in the same cluster
            numFaces += faces[0] && clusterH == clusterBiMap.getClusterH(faces[0].unwrap()) ? 1 : 0;
            numFaces += faces[1] && clusterH == clusterBiMap.getClusterH(faces[1].unwrap()) ? 1 : 0;

            // if there is exactly one face, the edge is a contour edge
            if (numFaces == 1)
            {
                // add the vertices of contour edges to an unordered set, which
                // automatically doesn't add duplicates
                for (auto vertexH : mesh.getVerticesOfEdge(edgeH))
                {
                    contourVertices.insert(vertexH);
                }
            }
        }
    }

    return std::vector<VertexHandle>(contourVertices.begin(), contourVertices.end());
}

template<typename BaseVecT>
BoundingRectangle<BaseVecT> calculateBoundingRectangle(
    const std::vector<VertexHandle>& contour,
    const BaseMesh<BaseVecT>& mesh,
    const Cluster<FaceHandle>& cluster,
    const FaceMap<Normal<BaseVecT>>& normals,
    float texelSize,
    ClusterHandle clusterH
)
{

    // TODO error handling for texelSize = 0
    // TODO reasonable error handling necessary for empty contour vector
    if (contour.size() == 0)
    {
        cout << "Empty contour array." << endl;
    }
    int minArea = std::numeric_limits<int>::max();

    float bestMinA, bestMaxA, bestMinB, bestMaxB;
    Vector<BaseVecT> bestBoundingAxisA, bestBoundingAxisB;

    // // calculate regression plane for the cluster
    Plane<BaseVecT> regressionPlane = calcRegressionPlane(mesh, cluster, normals);

    // support vector for the plane
    Vector<BaseVecT> supportVector = regressionPlane.project(mesh.getVertexPosition(contour[0]));

    // calculate two orthogonal vectors in the plane
    auto normal = regressionPlane.normal;
    auto pointInPlane = regressionPlane.project(mesh.getVertexPosition(contour[1])).asVector();
    auto boudningAxis1 = (pointInPlane - supportVector).cross(normal.asVector());
    boudningAxis1.normalize();

    Vector<BaseVecT> boundingAxis2 = boudningAxis1.cross(normal.asVector());
    boundingAxis2.normalize();

    // const float pi = boost::math::constants::pi<float>(); // FIXME: doesnt seem to work with c++11
    const float pi = std::atan(1) * 4; // reasonable approximation for pi

    // resolution of iterative improvement steps for a fourth rotation
    const float delta = (pi / 2) / 90;

    for(float theta = 0; theta < M_PI / 2; theta += delta)
    {
        // rotate the bounding box
        boudningAxis1 = boudningAxis1 * cos(theta) + boundingAxis2 * sin(theta);
        boudningAxis1.normalize();
        boundingAxis2 = boudningAxis1.cross(normal.asVector());
        boundingAxis2.normalize();

        // FIXME
        // calculate hessian normal forms for both planes to which the distances will be calculated

        // assume each bounding box axis is the normal of a plane, then the dot
        // product with the support vector is the support vectors negative
        // distance to the plane given by the axis (n * sv = -p)
        // Note that in contrast to the usual plane equations, the plane
        // distance is missing the negative sign
        const float planeDist1 = boudningAxis1.dot(supportVector);
        const float planeDist2 = boundingAxis2.dot(supportVector);


        float minDistA = std::numeric_limits<float>::max();
        float maxDistA = std::numeric_limits<float>::lowest();
        float minDistB = std::numeric_limits<float>::max();
        float maxDistB = std::numeric_limits<float>::lowest();


        // calculate the bounding box

        for(const auto contourVertexH : contour)
        {
            // TODO: Besser vorberechnen?
            //auto contourPoint = mesh.getVertexPosition(contourVertexH);
            // TODO: project nötig?
            auto contourPoint = regressionPlane.project(mesh.getVertexPosition(contourVertexH));

            // use hessian plane form for distance calculation
            // note the negative sign of planeDist*, since the calculation above
            // actually computes the negative distance
            // calculate distance to plane1
            float distA = boudningAxis1.dot(contourPoint) - planeDist1;
            // calculate distance to plane2
            float distB = boundingAxis2.dot(contourPoint) - planeDist2;

            // memorize largest positive and negative distance to both planes
            if (distA > maxDistA)
            {
                maxDistA = distA;
            }
            if (distA < minDistA)
            {
                minDistA = distA;
            }
            if (distB > maxDistB)
            {
                maxDistB = distB;
            }
            if (distB < minDistB)
            {
                minDistB = distB;
            }
        }

        // calculate predicted number of texels for both dimesions
        int texelsX = std::ceil((maxDistA - minDistA) / texelSize);
        int texelsY = std::ceil((maxDistB - minDistB) / texelSize);

        // iterative improvement of the area
        if(texelsX * texelsY < minArea)
        {
            minArea           = texelsX * texelsY;
            bestMinA          = minDistA;
            bestMaxA          = maxDistA;
            bestMinB          = minDistB;
            bestMaxB          = maxDistB;
            bestBoundingAxisA = boudningAxis1;
            bestBoundingAxisB = boundingAxis2;
        }
    }

    return BoundingRectangle<BaseVecT>(
        supportVector,
        bestBoundingAxisA,
        bestBoundingAxisB,
        normal,
        bestMinA,
        bestMaxA,
        bestMinB,
        bestMaxB
    );

}


} // namespace lvr2
