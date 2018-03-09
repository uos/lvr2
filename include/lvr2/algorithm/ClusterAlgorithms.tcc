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
 * ClusterAlgorithms.tcc
 *
 * @date 24.07.2017
 * @author Johan M. von Behren <johan@vonbehren.eu>
 * @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
 * @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
 * @author Rasmus Diederichsen <rdiederichse@uni-osnabrueck.de>
 */

#include <lvr2/algorithm/ContourAlgorithms.hpp>
#include <lvr2/algorithm/Planar.hpp>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>

#include <cmath>
#include <limits>
#include <unordered_set>


namespace lvr2
{

template<typename BaseVecT>
void removeDanglingCluster(BaseMesh<BaseVecT>& mesh, size_t sizeThreshold)
{
    // Do cluster growing without a predicate, so cluster will consist of connected faces
    auto clusterSet = clusterGrowing(mesh, [](auto referenceFaceH, auto currentFaceH)
    {
        return true;
    });

    // Remove all faces in too small clusters
    for (auto clusterH: clusterSet)
    {
        auto cluster = clusterSet.getCluster(clusterH);
        if (cluster.handles.size() < sizeThreshold)
        {
            for (auto faceH: cluster.handles)
            {
                mesh.removeFace(faceH);
            }
        }
    }
}

template<typename BaseVecT>
vector<vector<VertexHandle>> findContours(
    BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& clusters,
    ClusterHandle clusterH
)
{
    auto cluster = clusters[clusterH];

    DenseVertexMap<bool> boundaryVertices(cluster.handles.size() * 3, false);
    vector<vector<VertexHandle>> allContours;
    // only used inside edge loop but initialized here to avoid heap allocations
    vector<VertexHandle> contour;

    for (auto faceH: cluster.handles)
    {
        for (auto edgeH: mesh.getEdgesOfFace(faceH))
        {
            auto faces = mesh.getFacesOfEdge(edgeH);
            if (faces[0] && faces[1])
            {
                auto otherFace = faces[0].unwrap();

                if (otherFace == faceH)
                {
                    otherFace = faces[1].unwrap();
                }

                // continue if other face is in same cluster
                if (clusters.getClusterOf(otherFace) &&
                    clusters.getClusterOf(otherFace).unwrap() == clusterH
                    )
                {
                    continue;
                }
            }


            auto vertices = mesh.getVerticesOfEdge(edgeH);

            // edge already in another boundary of this cluster
            if (boundaryVertices[vertices[0]] || boundaryVertices[vertices[1]])
            {
                continue;
            }

            contour.clear();
            calcContourVertices(mesh, edgeH, contour, [clusters, clusterH](auto fH)
            {
                auto c = clusters.getClusterOf(fH);

                // return true if current face is in this cluster
                return c && c.unwrap() == clusterH;
            });

            allContours.push_back(contour);

            // mark all vertices we got back as visited
            for (auto vertexH: contour)
            {
                boundaryVertices[vertexH] = true;
            }
        }

    }

    return allContours;
}

template<typename BaseVecT>
vector<VertexHandle>
simplifyContour(const BaseMesh<BaseVecT>& mesh, const vector<VertexHandle>& contour, float threshold) {
    auto out = vector<VertexHandle>();

    // first point is always part of the simplified contour
    out.push_back(contour[0]);

    // current point
    auto p0 = mesh.getVertexPosition(contour[0]);
    // next point after first point
    auto p1 = mesh.getVertexPosition(contour[1]);

    // previous test point handle
    auto piH = contour[1];
    // current test point handle
    auto pjH = piH;

    for (size_t i = 2 ; i < contour.size(); ++i)
    {
        piH = pjH;
        pjH = contour[i];

        // current key vector / line to check against
        auto vec = p1 - p0;
        vec.normalize();

        // vector of next and after next point
        auto vec2 = mesh.getVertexPosition(pjH) - mesh.getVertexPosition(piH);
        vec2.normalize();

        if (vec.dot(vec2) >= threshold)
        {
            continue;
        }

        // found next point in line
        out.push_back(piH);

        // define new line to check against
        p0 = mesh.getVertexPosition(piH);
        p1 = mesh.getVertexPosition(pjH);
    }

    // also add last last point to simplified contour
    out.push_back(pjH);

    return out;
}


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
