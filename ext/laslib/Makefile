all:
	cd src && make
	cd example && make
	cd test && make

clean:
	cd src && make clean
	cd example && make clean
	cd test && clean
	rm -f lib/*
	rm -f libD/*

clobber:
	cd src && make clobber
	cd example && make clobber
	cd test && make clobber
	rm -f lib/*
	rm -f libD/*
