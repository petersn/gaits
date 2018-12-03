#!/usr/bin/python

import simulation


world = simulation.PhysicsWorld()
state = simulation.ObjectState()
state.xyz.y = 10
box = simulation.Box(world, simulation.Vec(1, 1, 1), state, 1.0)
state.xyz.y = 1
state.xyz.x = 1.1
state.xyz.z = 0.12
box2 = simulation.Box(world, simulation.Vec(1, 1, 1), state, 1.0)
plane = simulation.StaticPlane(world, simulation.Vec(0, 1, 0), 0)

for _ in xrange(100):
	world.step(0.03, 60)
	box.get_data(state)
	print state.xyz.y
	#print to_str(state)

