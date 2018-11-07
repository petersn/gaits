#!/usr/bin/python

import simulation

def to_str(x):
	if isinstance(x, simulation.Vec):
		return "<%f,%f,%f,%f>" % (x.x, x.y, x.z, x.w)
	elif isinstance(x, simulation.ObjectState):
		return "<pos=%s vel=%s rot=%s>" % (to_str(x.xyz), to_str(x.vel), to_str(x.quat))
	raise NotImplementedError

world = simulation.PhysicsWorld()
state = simulation.ObjectState()
state.xyz.y = 10
box = simulation.Box(world, simulation.Vec(1, 1, 1), state, 1.0)
state.xyz.y = 1
state.xyz.x = 0.1
state.xyz.z = 0.12
box2 = simulation.Box(world, simulation.Vec(1, 1, 1), state, 1.0)
plane = simulation.StaticPlane(world, simulation.Vec(0, 1, 0), 0)

for _ in xrange(100):
	world.step(0.03, 60)
	box.get_data(state)
	print state.xyz.y
	#print to_str(state)

