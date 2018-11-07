
CXXFLAGS=-g `sdl-config --cflags` -fPIC -std=c++11 -pthread -I/usr/include/bullet -Wall -Wextra -Wpedantic -Wno-sign-compare `pkg-config --cflags --libs python2` -fdiagnostics-color
LIBS=`sdl-config --libs` -lSDL_mixer -lGL -lGLU -lpng -lSDL -lBulletDynamics -lBulletCollision -lLinearMath
CXX=g++

all: _simulation.so

simulation_wrap.cxx: simulation.h simulation.i Makefile
	swig -c++ -python simulation.i

%.o: %.cxx
	$(CXX) -c $(CXXFLAGS) $^ -o $@

_simulation.so: simulation.o simulation_wrap.o
	$(CXX) -shared -Wl,-soname,$@ -o $@ $^ $(LIBS)

.PHONY: clean
clean:
	rm -f *.o *.pyc simulation_wrap.cxx _simulation.so

