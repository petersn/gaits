
CXXFLAGS=-g  -fPIC -std=c++14 -pthread  -Wall -Wextra -Wpedantic -Wno-sign-compare  -fdiagnostics-color
CXXFLAGS+=`pkg-config --cflags --libs python2`
CXXFLAGS+=`sdl-config --cflags`
CXXFLAGS+=-I/usr/include/bullet
LIBS=`sdl-config --libs` `pkg-config --libs python2` -lSDL_mixer -lGL -lGLU -lpng -lSDL -lBulletDynamics -lBulletCollision -lLinearMath
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

