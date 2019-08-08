hyperfine -r 20 "./lvr2_registration hannover/ -s 1 -e 200 -d 75 -i 100" "./lvr2_registration hannover/ -s 1 -e 2
00 -d 75 -i 100 -r 340" "./slam6D hannover/ -q -s 1 -e 200 -d 75 -i 100" "./slam6D hannover/ -q -s 1 -e 200 -d 75 -i 100
 -r 10" "./lvr2_registration hannover_bin/ -f \"scan%03i.ply\" -s 1 -e 200 -d 75 -i 100" "./lvr2_registration hannover_b
in/ -f \"scan%03i.ply\" -s 1 -e 200 -d 75 -i 100 -r 340"