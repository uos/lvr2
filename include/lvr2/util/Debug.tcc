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
 * Debug.tcc
 *
 *  @date 18.07.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#include <unordered_map>

using std::unordered_map;

#include <lvr2/algorithm/Materializer.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/algorithm/FinalizeAlgorithms.hpp>


namespace lvr2
{

template<typename BaseVecT>
void writeDebugMesh(
    const BaseMesh<BaseVecT>& mesh,
    string filename,
    Rgb8Color color
)
{
    // Generate color map
    // TODO: replace with better impl of attr map
    DenseVertexMap<Rgb8Color> colorMap;
    colorMap.reserve(mesh.numVertices());

    for (auto vH: mesh.vertices())
    {
        colorMap.insert(vH, color);
    }

    // Set color to mesh
    SimpleFinalizer<BaseVecT> finalize;
    finalize.setColorData(colorMap);

    // Get buffer
    auto buffer = finalize.apply(mesh);

    // Save mesh
//    auto m = boost::make_shared<Model>(buffer);
//    ModelFactory::saveModel(m, filename);
}

template<typename BaseVecT>
vector<vector<VertexHandle>> getDuplicateVertices(const BaseMesh<BaseVecT>& mesh)
{
    // Save vertex handles "behind" equal points
    unordered_map<Vector<BaseVecT>, vector<VertexHandle>> uniquePoints;
    for (auto vH: mesh.vertices())
    {
        auto point = mesh.getVertexPosition(vH);
        uniquePoints[point].push_back(vH);
    }

    // Extract all vertex handles, where one point has more than one vertex handle
    vector<vector<VertexHandle>> duplicateVertices;
    for (auto elem: uniquePoints)
    {
        auto vec = elem.second;
        if (vec.size() > 1)
        {
            duplicateVertices.push_back(vec);
        }
    }

    return duplicateVertices;
}

template<typename BaseVecT>
void
writeDebugContourMesh(
    const BaseMesh<BaseVecT>& mesh,
    string filename,
    Rgb8Color connectedColor,
    Rgb8Color contourColor,
    Rgb8Color bugColor
)
{
    DenseVertexMap<Rgb8Color> color_vertices(mesh.numVertices(), connectedColor);
    for (auto eH: mesh.edges())
    {
        auto vertices = mesh.getVerticesOfEdge(eH);

        // found a contour
        auto numFaces = mesh.numAdjacentFaces(eH);
        if (1 == numFaces)
        {
            color_vertices[vertices[0]] = contourColor;
            color_vertices[vertices[1]] = contourColor;
        }
            // found something weird
        else if (2 != numFaces)
        {
            color_vertices[vertices[0]] = bugColor;
            color_vertices[vertices[1]] = bugColor;
        }
    }

    // Finalize mesh (convert it to simple `MeshBuffer`)
    SimpleFinalizer<BaseVecT> finalize;
    finalize.setColorData(color_vertices);
    auto buffer = finalize.apply(mesh);

    // Save mesh
    cout << "IMPLEMENT ME " << endl;

    //auto m = boost::make_shared<Model>(buffer);
    //ModelFactory::saveModel(m, filename);
}

} // namespace lvr2
