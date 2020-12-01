^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package lvr2
^^^^^^^^^^^^^^^^^^^^^^^^^^

20.11.1 (2020-11-11)
--------------------
* fix for vtk8 issue
* add missing include and clean up using std
* use double for precision and refactor cl sor tool
* adds sor filter tool based on gpu knn
* added hdf5features example
* faster loading. map buffers instead of insert next cells
* added embree dep to cmake.in
* merged raycaster and cleanup
* working OpenCL Raycaster with better structure
* fix colors in filtering
* some performance optimizations
* migrated dmc approach

20.7.1 (2020-07-09)
-------------------
* removed cl2.hpp, FeatureProjector.cpp from lib
* added meta information factory with slam6d and yaml support
* add debian folder from deb-lvr2 repo.
* add read vertex normals and colors from MeshBuffer, e.g., ply files
* add std arrays to conversion proxy for attribute map to channel conversion
* the large scale reconstruction chunk size is now dictated by the chunk manager
* make most things configurable for chunked mesh visualization
* hyperspectral meta information can be saved and loaded
* implemented basic load and save function of new IO features
* added function to save the tsdf-values in the chunk manager
* switched to new hdf5 io scheme for chunks and added a chunking pipeline with a multiple layer support
* integrated unoptimized hdf5-input for partial reconstruction
* fully integrated existing approach of dmc
* Contributors: Alexander Mock, Bao Tran, Benedikt Schumacher, Kevin Rüter, Lennart Niecksch, Malte kl. Piening, Marcel Wiegand, Raphael Marx, Sebastian Pütz, Thomas Wiemann, Timo Osterkamp, Wilko Müller

19.12.1 (2020-01-04)
--------------------
* Initial release of lvr2
* Contributors: = Adrien Devresse, Aleksandr Ovcharenko, Alexander Altemöller, Alexander Löhr, Alexander Mock, Ali Can Demiralp, Ann-Katrin Haeuser, Bao Tran, Benedikt Schumacher, Christian Swan, Daniel Nachbaur, David J. Luitz, Denis Meyer, Devresse Adrien, Dmitri Bichko, Dominik Feldschnieders, Felix Igelbrink, Fernando Pereira, Florian Otte, Henning Deeken, Henning Strüber, Isaak Mitschke, Jan Philipp Vogtherr, Jan Toennemann, Jochen Sprickerhof, Johan M. von Behren, Juan Hernando Vieites, Kim Oliver Rinnewitz, Kristin Schmidt, Lars Kiesow, Lennart Niecksch, Lukas Kalbertodt, Malte kl. Piening, Marcel Mrozinski, Marcel Wiegand, Martin Günther, Matthias Greshake, Michael Görner, Michael V. DePalatis, Mike Gevaert, Nils Niemann, Pablo Toharia, Raphael Marx, Rasmus Diederichsen, Sabine Rast, Sebastian Pütz, Sergei Sobol, Simon Herkenhoff, Stefan Eilemann, Steffen Hinderink, Sven Albrecht, Thomas Wiemann, Tristan Igelbrink, Wilko Müller, Wolf Vollprecht

