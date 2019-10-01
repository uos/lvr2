# LargeScale Reconstruction: Usage

after compiling use the following command to execute the standard ls-reconstruction:

```bash
./bin/lvr2_largescale_reconstruct /pointcloud.ply
```

The standard ls-reconstruction will use a Kd-tree structure to subdivide the 
bounding box of a given pointcloud in sub-boxes depending on the number of points. The resulting Sub-Boxes will be 
reconstructed individually and then merged to a bigger mesh.

To improve the reconstruction time, use the following option:

 ```bash
 ./bin/lvr2_largescale_reconstruct /pointcloud.ply --useGPU
 ```

## Largescale Reconstruction: VirtualGrid

to use a grid-based method to subdivide the pointcloud, use the following command:

```bash
./bin/lvr2_largescale_reconstruct /pointcloud.ply --useVGrid=1
```

This will divide the pointcloud according to a specific grid 
(standard cell-size: 2000x2000x2000) relative to the origin (0,0,0).

```bash
______|______|______
      |    . |   . .  
      |   .  |   .   
______|____._|______
      |    . |  .    
      |  . . |   .   
______|______|______
      |      |      
      |      |      
______0______|______
```
Only the chunks which contains a certain amount of points will be kept for the reconstruction.


# Partial Reconstruction: Usage
Partial Reconstruction allows simple expansion of a given mesh. 

Requirement:
* initial Mesh was created via virtual-grid method (see above)
* a global PointCloud (**pointcloud_global.ply**) exists (PointCloud of initial mesh + new PointCloud)
* the new PointCloud to be inserted is registered accordingly

To create a new Mesh from a PointCloud (**pointcloud_new.ply**) and expand a given one (**bigMesh.ply**), 
execute the following command:

```bash
./bin/lvr2_largescale_reconstruct ./pointcloud_global.ply -- partialReconstruct=./pointcloud_new.ply --useVGrid=1
``` 

The user has to ensure, that the same chunksize was used in both reconstruction process.
