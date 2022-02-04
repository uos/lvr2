
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
    timeline: false,
    infoBox: false,
    animation: false,
    scene3DOnly: true,
    imageryProvider: new Cesium.GridImageryProvider({
        tilingScheme: new TilingScheme(),
    }),
});
window['viewer'] = viewer;

var args = new URLSearchParams(window.location.search);
var url;
if (args.has('url')) {
    url = args.get('url');
} else {
    url = "/build/chunk.3dtiles";
}
if (!url.endsWith("tileset.json")) {
    if (!url.endsWith("/")) {
        url += "/";
    }
    url += "tileset.json";
}

var tileset = viewer.scene.primitives.add(
    new Cesium.Cesium3DTileset({
        url,
        debugShowBoundingVolume: true,
        // debugShowRenderingStatistics: true,
        // debugShowMemoryUsage: true,
        backFaceCulling: false,
    })
);

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
