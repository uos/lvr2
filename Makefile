
END = 75
Q = 
D = 0.75

compile: build
	make -s -C build -j4 lvr2_registration

build:
	mkdir build
	cd build && cmake -DCMAKE_BUILD_TYPE=Release ..; cd ..

clean:
	rm -rf build

run:
	build/bin/lvr2_registration ../obj/hannover_lvr -s 1 -e $(END) -i 100 -d 75 $(Q)

run_slam:
	cd ../obj/slam6d && bin/slam6D ../hannover_slam6d -s 1 -e $(END) -i 100 -d 75 $(Q)

show:
	../obj/slam6d/bin/show ../obj/hannover_lvr --no-fog -s 1 -e $(END)

show_slam:
	../obj/slam6d/bin/show ../obj/hannover_slam6d --no-fog -s 1 -e $(END)

clear_frames:
	rm ../obj/hannover_lvr/*.frames ../obj/hannover_slam6d/*.frames

run_hase:
	build/bin/lvr2_registration /data/haseschacht -f ply --pose-format dat -s 1 -e 2 -d $(D) $(Q)
