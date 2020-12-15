Tutorial on how to use the Ground Level Extractor for DTM Creation

This tool can extract a DTM from a PLY point cloud and generate a texture based on the collected data. Additionally, it can transform the model into a georeferenced coordinate system via reference points. When the user provides a GeoTIFF file that fits the recorded area, the GeoTIFF's bands can be projected on the model. If alternatively, no GeoTIFF is provided, a texture that showcases the height difference between the original point cloud and model is generated.

At this point, it is not possible to export the models in their fully transformed form. The programme only applies the rotation.The translation output is saved in an additional .txt file.

To gain an overview of the different options, use the --help command when calling the ground level extractor. 
The most important parameters are summarised here:
--inputFile: necessary for the program to work. Path to the file that contains the point cloud data.
--outputFile: the name of the OBJ and PLY file that will be created.
--resolution: sets the distance between nodes, and thus also the number of points. Lowering this parameter increases the number of points.

Tutorial Files:

Example 1.1:
Generating a model based on a PLY point cloud (no GeoTIFF and no reference points) using Nearest Neighbor.

command:
bin/lvr2_ground_level_extractor --inputFile ~/Files/trees.ply --extractionMethod NN --numberNeighbors 2

To run the programme, a point cloud has to be provided. The user can choose between one of three extraction methods: NN(Nearest Neighbor, the default option), IMA(Improved Moving Average) and THM(Threshold Method). This generates a PLY file and an OBJ file. The latter not only depicts the model, but also shows the height difference between the model and the original point cloud as projected texture.

The method Nearest Neighbor uses the points surrounding the nodes to compute their height. The amount of utilized neighbours  can be adjusted with the command --numberNeighbors. If a node does not have a sufficient amount of near points, it is excluded from the model.

Example 1.2:
Generating a model based on a PLY point cloud (no GeoTIFF and no reference points) using Improved Moving Average

command:
bin/lvr2_ground_level_extractor --inputFile ~/Files/trees.ply --extractionMethod IMA --numberNeighbors 2 --minRadius 0.5 --maxRadius 2 --radiusSteps 50

The method Improved Moving Average searches for points in a pre-defined radius. If not enough points are found, the radius is expanded until the maximum radius is reached. When the number is insufficient after reaching the maximum radius, the node is left blank.

Example 1.3:
Generating a model based on a PLY point cloud (no GeoTIFF and no reference points) using the Threshold Method

command:
bin/lvr2_ground_level_extractor --inputFile ~/Files/trees.ply --extractionMethod THM --swSize 3 --swThreshold 1 --lwSize 51 --lwThreshold 3 --slopeThreshold 30 --resolution 0.5

The Threshold Method excludes non-ground nodes from the model by testing them in three different ways: Small Window Thresholding (removes small objects), Slope Thresholding (removes very small objects and noise) and Large Window Thresholding (removes large objects). These tests’ parameters need to be set according to the target terrain. The window thresholding method compares nodes to their surrounding nodes in a user-defined area and it computes the difference between the lowest recorded height value in the area and the observed node. If this difference exceeds a threshold, the node is excluded from the model. Slope Thresholding calculates the angle of the slope between two neighbouring nodes. If this angle breaks the user-defined threshold, the node is excluded.

Example 2:
Generating a georeferenced Model and projecting a GeoTIFF on it.

command:
bin/lvr2_ground_level_extractor --inputFile ~/Files/ggreens.ply --inputReferencePairs coordinates.txt --inputGeoTIFF ~/Files/ggreens.tif --extractionMethod THM --swSize 3 --swThreshold 0.5 --lwSize 21 --lwThreshold 2 --slopeThreshold 30 --resolution 0.5

This input will create a georeferenced model with the content of the first GeoTIFF band rendered on it. Which bands are used can be changed with the commands --startingBand and --numberOfBands. At the current time, it is not possible to rotate and translate the model according to the matrix, since the output of coordinates as doubles is not possible. The translation of the coordinates is not calculated. Because of this, the translation and rotation matrices are written on a file, so the missing translation can be added manually, if necessary.

Example 3:
Example 2 + changing the target coordinate system

command:
bin/lvr2_ground_level_extractor --inputFile ~/Files/ggreens.ply --inputReferencePairs coordinates.txt --inputGeoTIFF ~/Files/ggreens.tif --extractionMethod THM --swSize 3 --swThreshold 0.5 --lwSize 21 --lwThreshold 2 --slopeThreshold 30 --targetSystem EPSG:3857 --resolution 0.75

EPSG:3857 marks the Web Mercator projection/Google Web Mercator and is a standard for Web mapping applications. It is used as an example system to transform the GeoTIFFs data into. Adding this argument to the programme call leads to the creation of a GeoTIFF containing the transformed data of the original GeoTIFF. The resulting GeoTIFF is applied for texture generation. Please note that the GeoTIFF file will be large (in this case ~2-3 GB). Accordingly, this procedure should only be carried out if sufficient space on the hard drive is available. The reference points are transformed to fit the new system as well.

Known Bugs:
- Target Coordinate System: some coordinate systems lead to the textures not being displayed correctly on the model. 



