/**
 * Copyright (c) 2022, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Tiles3dIO.tcc
 *
 * @date   01.07.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include "Tiles3dIO.hpp"

#include "lvr2/io/modelio/B3dmIO.hpp"

#include <boost/filesystem.hpp>

#include <Cesium3DTiles/Tileset.h>

namespace lvr2
{

// functions that don't depend on the template parameter and can this be moved to Tiles3dIO.cpp
namespace Tiles3dIO_internal
{

void convertBoundingBox(const pmp::BoundingBox& in, Cesium3DTiles::BoundingVolume& out);
void indexToName(int i, std::string& name, size_t max);
void writeTileset(Cesium3DTiles::Tileset& tileset, const std::string& outputDir, float scale);

}

template<typename BaseVecT>
Tiles3dIO<BaseVecT>::Tiles3dIO(const std::string& directory)
    : m_rootDir(directory)
{
    if (m_rootDir.back() != '/')
    {
        m_rootDir += "/";
    }
    if (boost::filesystem::exists(m_rootDir))
    {
        boost::filesystem::remove_all(m_rootDir);
    }
    boost::filesystem::create_directories(m_rootDir);
    boost::filesystem::create_directory(m_rootDir + "tiles/");
}

template<typename BaseVecT>
void Tiles3dIO<BaseVecT>::write(TreeConstPtr& tree, float scale)
{
    Cesium3DTiles::Tileset tileset;

    writeTiles(tileset.root, tree, m_rootDir, "tiles/s");

    Tiles3dIO_internal::writeTileset(tileset, m_rootDir, scale);
}

template<typename BaseVecT>
void Tiles3dIO<BaseVecT>::writeTiles(Cesium3DTiles::Tile& tile,
                                     TreeConstPtr& tree,
                                     const std::string& outputDir,
                                     const std::string& prefix)
{
    tile.geometricError = tree->depth() == 0 ? 0.0 : std::pow(10, tree->depth() - 1);
    Tiles3dIO_internal::convertBoundingBox(tree->bb(), tile.boundingVolume);

    auto& children = tree->children();
    tile.children.resize(children.size());
    for (size_t i = 0; i < children.size(); i++)
    {
        std::string next_prefix = prefix;
        Tiles3dIO_internal::indexToName(i, next_prefix, children.size() - 1);
        writeTiles(tile.children[i], children[i], outputDir, next_prefix);
    }

    auto mesh = tree->mesh();
    if (mesh)
    {
        std::string filename = prefix;
        if (!tree->isLeaf())
        {
            Tiles3dIO_internal::indexToName(-1, filename, children.size() - 1);
        }
        filename += ".b3dm";

        Cesium3DTiles::Content content;
        content.uri = filename;
        tile.content = content;
        tile.refine = Cesium3DTiles::Tile::Refine::REPLACE;

        auto model = std::make_shared<Model>();
        auto pmp_mesh = mesh->get();
        model->m_mesh = pmp_mesh->toMeshBuffer();
        pmp_mesh.reset();

        B3dmIO io;
        io.setModel(model);
        // if (tree->isLeaf()) // TODO: add compression parameter
        // {
        //     io.save(outputDir + filename);
        // }
        // else
        // {
            io.saveCompressed(outputDir + filename);
        // }
    }
}

} // namespace lvr2
