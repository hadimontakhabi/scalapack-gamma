all:
	mpicc -o scalapack-gamma -g -O2 scalapack-gamma.c -lscalapack-openmpi -lblacsCinit-openmpi -lblacs-openmpi -llapack -lblas -lgfortran

clean:
	rm -rf *~ *# scalapack-gamma
