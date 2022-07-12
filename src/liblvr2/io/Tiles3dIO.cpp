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
 * Tiles3dIO.cpp
 *
 * @date   01.07.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "lvr2/io/Tiles3dIO.hpp"

#include <Cesium3DTilesWriter/TilesetWriter.h>

extern const char* VIEWER_HTML;

namespace lvr2::Tiles3dIO_internal
{

void convertBoundingBox(const pmp::BoundingBox& in, Cesium3DTiles::BoundingVolume& out)
{
    auto center = in.center();
    auto halfVector = in.max() - center;
    out.box =
    {
        center.x(), center.y(), center.z(),
        halfVector.x(), 0, 0,
        0, halfVector.y(), 0,
        0, 0, halfVector.z()
    };
}

void indexToName(int i, std::string& name, size_t max)
{
    constexpr size_t RADIX = 10 + 26 + 26;
    if (i < 0)
    {
        // meta segment
        do
        {
            name += '_';
            max /= RADIX;
        }
        while (max > 0);

        return;
    }
    if (max > RADIX)
    {
        indexToName(i / RADIX, name, max / RADIX);
        i %= RADIX;
    }
    name += (i < 10 ? '0' + i : (i < 36 ? 'a' + i - 10 : 'A' + i - 36));
}

void writeTileset(Cesium3DTiles::Tileset& tileset, const std::string& outputDir, float scale)
{
    tileset.asset.version = "1.0";
    tileset.geometricError = 1e6; // tileset should always be rendered -> set error very high
    // note: geometricError on Tileset does the opposite of the one on Tile


    auto& root = tileset.root;
    root.refine = Cesium3DTiles::Tile::Refine::ADD;

    double minZ = root.boundingVolume.box[2] - root.boundingVolume.box.back(); // see convertBoundingBox: center.z() - halfVector.z()
    root.transform =
    {
        // 4x4 matrix to place the object somewhere on the globe
        -scale, 0, 0, 0,
        0, 0, scale, 0,
        0, scale, 0, 0,
        0, 6378137 - minZ * scale, 0, 1
    };

    Cesium3DTilesWriter::TilesetWriter writer;
    auto result = writer.writeTileset(tileset);

    if (!result.warnings.empty())
    {
        std::cerr << "Warnings writing tileset: " << std::endl;
        for (auto& e : result.warnings)
        {
            std::cerr << e << std::endl;
        }
    }
    if (!result.errors.empty())
    {
        std::cerr << "Errors writing tileset: " << std::endl;
        for (auto& e : result.errors)
        {
            std::cerr << e << std::endl;
        }
        throw std::runtime_error("Error writing tileset");
    }

    std::string tileset_file = outputDir + "tileset.json";
    std::cout << timestamp << "Writing " << tileset_file << std::endl;

    std::ofstream tileset_out(tileset_file, std::ios::binary);
    tileset_out.write((char*)result.tilesetBytes.data(), result.tilesetBytes.size());
    tileset_out.close();

    std::string viewer_file = outputDir + "index.html";
    std::cout << timestamp << "Writing " << viewer_file << std::endl;

    std::ofstream viewer_out(viewer_file);
    viewer_out << VIEWER_HTML;
    viewer_out.close();
}

} // namespace lvr2::Tiles3dIO_internal



const char* VIEWER_HTML = R"=======(
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/svg+xml" href="https://cesium.com/cesium-logomark.svg">
    <link rel="icon" type="image/png" sizes="192x192" href="https://cesium.com/cesium-logomark-192.png">
    <title>3D Tiles Viewer</title>

    <script src="https://cesium.com/downloads/cesiumjs/releases/1.91/Build/Cesium/Cesium.js"></script>
    <link href="https://cesium.com/downloads/cesiumjs/releases/1.91/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
    <style>
        #cesiumContainer {
            display: block;
            position: absolute;
            top: 0;
            left: 0;
            border: none;
            width: 100%;
            height: 100%;
        }
        body {
            margin: 0;
            padding: 0;
        }
    </style>
</head>

<body>
    <div id="cesiumContainer"></div>

    <script>
        class TilingScheme extends Cesium.GeographicTilingScheme {
            getNumberOfXTilesAtLevel(level) {
                return super.getNumberOfXTilesAtLevel(Math.min(level, 10));
            }
            getNumberOfYTilesAtLevel(level) {
                return super.getNumberOfYTilesAtLevel(Math.min(level, 10));
            }
        }

        var viewer = new Cesium.Viewer("cesiumContainer", {
            vrButton: true,
            sceneModePicker: false,
            baseLayerPicker: false,
            geocoder: false,
            infoBox: false,
            animation: false,
            scene3DOnly: true,
            imageryProvider: new Cesium.GridImageryProvider({
                tilingScheme: new TilingScheme(),
            }),
        });
        window['viewer'] = viewer;

        var tileset = viewer.scene.primitives.add(
            new Cesium.Cesium3DTileset({ url: "tileset.json" })
        );
        window['tileset'] = tileset;

        tileset.lightColor = new Cesium.Cartesian3(3, 2.8, 2.4);

        tileset.readyPromise
            .then(function () {
                viewer.zoomTo(tileset);
                viewer.homeButton.viewModel.command.beforeExecute.addEventListener(function (e) {
                    viewer.zoomTo(tileset);
                    e.cancel = true;
                });
            })
            .otherwise(function (error) {
                throw error;
            });

        function freeze() {
            tileset.debugFreezeFrame = !tileset.debugFreezeFrame;
        }
        function colors() {
            tileset.debugColorizeTiles = !tileset.debugColorizeTiles;
        }
        function bb() {
            tileset.debugShowBoundingVolume = !tileset.debugShowBoundingVolume;
        }
        function stats() {
            tileset.debugShowStatistics = !tileset.debugShowStatistics;
        }
    </script>
</body>
</html>
)=======";
