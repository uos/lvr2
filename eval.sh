#!/usr/bin/sh

img_x=7479
img_y=900
path=~/tmp/hyperspectral

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 10 --hsp_chunk_1 10 --hsp_chunk_2 10
du hyper.h5 >> all_dims.txt
echo "10 " >> all_dims.txt
rm hyper.h5


bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 20 --hsp_chunk_1 20 --hsp_chunk_2 20
du hyper.h5 >> all_dims.txt
echo "20 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 30 --hsp_chunk_1 30 --hsp_chunk_2 30
du hyper.h5 >> all_dims.txt
echo "30 " >> all_dims.txt
rm hyper.h5


bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 40 --hsp_chunk_1 40 --hsp_chunk_2 40
du hyper.h5 >> all_dims.txt
echo "40 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 50 --hsp_chunk_1 50 --hsp_chunk_2 50
du hyper.h5 >> all_dims.txt
echo "50 " >> all_dims.txt
rm hyper.h5


bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 60 --hsp_chunk_1 60 --hsp_chunk_2 60
du hyper.h5 >> all_dims.txt
echo "60 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 70 --hsp_chunk_1 70 --hsp_chunk_2 70
du hyper.h5 >> all_dims.txt
echo "70 " >> all_dims.txt
rm hyper.h5


bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 80 --hsp_chunk_1 80 --hsp_chunk_2 80
du hyper.h5 >> all_dims.txt
echo "80 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 90 --hsp_chunk_1 90 --hsp_chunk_2 90
du hyper.h5 >> all_dims.txt
echo "90 " >> all_dims.txt
rm hyper.h5


bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 100 --hsp_chunk_1 100 --hsp_chunk_2 100
du hyper.h5 >> all_dims.txt
echo "100 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 200 --hsp_chunk_1 200 --hsp_chunk_2 200
du hyper.h5 >> all_dims.txt
echo "200 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 300 --hsp_chunk_1 300 --hsp_chunk_2 300
du hyper.h5 >> all_dims.txt
echo "300 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 400 --hsp_chunk_1 400 --hsp_chunk_2 400
du hyper.h5 >> all_dims.txt
echo "400 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 500 --hsp_chunk_1 500 --hsp_chunk_2 500
du hyper.h5 >> all_dims.txt
echo "500 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 600 --hsp_chunk_1 600 --hsp_chunk_2 600
du hyper.h5 >> all_dims.txt
echo "600 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 700 --hsp_chunk_1 700 --hsp_chunk_2 700
du hyper.h5 >> all_dims.txt
echo "700 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 800 --hsp_chunk_1 800 --hsp_chunk_2 800
du hyper.h5 >> all_dims.txt
echo "800 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 900 --hsp_chunk_1 900 --hsp_chunk_2 900
du hyper.h5 >> all_dims.txt
echo "1000 " >> all_dims.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 1000 --hsp_chunk_1 1000 --hsp_chunk_2 1000
du hyper.h5 >> all_dims.txt
echo "1000 " >> all_dims.txt
rm hyper.h5

###############################################

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 1 --hsp_chunk_1 $img_y --hsp_chunk_2 $img_x
du hyper.h5 >> images.txt
echo "1 " >> images.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 2 --hsp_chunk_1 $img_y --hsp_chunk_2 $img_x
du hyper.h5 >> images.txt
echo "2 " >> images.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 3 --hsp_chunk_1 $img_y --hsp_chunk_2 $img_x
du hyper.h5 >> images.txt
echo "3 " >> images.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 4 --hsp_chunk_1 $img_y --hsp_chunk_2 $img_x
du hyper.h5 >> images.txt
echo "4 " >> images.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 5 --hsp_chunk_1 $img_y --hsp_chunk_2 $img_x
du hyper.h5 >> images.txt
echo "5 " >> images.txt
rm hyper.h5

##########################################################################################

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 1 --hsp_chunk_1 50 --hsp_chunk_2 50
du hyper.h5 >> slices.txt
echo "50 " >> slices.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 1 --hsp_chunk_1 100 --hsp_chunk_2 100
du hyper.h5 >> slices.txt
echo "100 " >> slices.txt
rm hyper.h5


bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 1 --hsp_chunk_1 200 --hsp_chunk_2 200
du hyper.h5 >> slices.txt
echo "200 " >> slices.txt
rm hyper.h5


bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 1 --hsp_chunk_1 300 --hsp_chunk_2 300
du hyper.h5 >> slices.txt
echo "300 " >> slices.txt
rm hyper.h5


bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 1 --hsp_chunk_1 400 --hsp_chunk_2 400
du hyper.h5 >> slices.txt
echo "400 " >> slices.txt
rm hyper.h5

bin/lvr2_hdf5tool --dataDir $path --hsp_chunk_0 1 --hsp_chunk_1 500 --hsp_chunk_2 500
du hyper.h5 >> slices.txt
echo "500 " >> slices.txt
rm hyper.h5

