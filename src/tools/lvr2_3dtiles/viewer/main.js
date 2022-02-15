
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
var url = args.get('url') || "/build/chunk.3dtiles";
if (!url.endsWith("tileset.json")) {
    if (!url.endsWith("/")) {
        url += "/";
    }
    url += "tileset.json";
}

var lower_args = new URLSearchParams(window.location.search.toLowerCase().replace(/[-_]/, ''));

var debug = ['debug', 'showdebug', 'debugdisplay'].some(x => lower_args.has(x));

var show_bb = debug || ['bb', 'boundingbox', 'showbb', 'showboundingbox'].some(x => lower_args.has(x));

var tileset = viewer.scene.primitives.add(
    new Cesium.Cesium3DTileset({
        url,
        debugShowBoundingVolume: show_bb,
        debugShowRenderingStatistics: debug,
        debugShowMemoryUsage: debug,
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
