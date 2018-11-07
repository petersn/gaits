#!/usr/bin/python
"""
physics.py

Provides a wrapper around simulation.py (which is the SWIG wrapper for _simulation.so), with a slightly higher level interface.
Includes facilities for serializing/deserializing the world state, for rendering in Blender.
"""

from __future__ import print_function
import numpy as np
import json
try:
	import simulation
except ImportError as e:
	class Dump: pass
	thing = Dump()
	class simulation:
		class DoNothing:
			def __init__(self, *args, **kwargs):
				pass
		class PhysicsWorld(DoNothing): pass
		class Box(DoNothing):
			def set_data(self, x): pass
		class StaticPlane(DoNothing): pass
		class Vec(DoNothing): pass
		class ObjectState(DoNothing):
			xyz = vel = quat = avel = thing

origin = np.array([0.0, 0.0, 0.0])
identity_quat = np.array([0.0, 0.0, 0.0, 1.0])

def to_str(x):
	if isinstance(x, simulation.Vec):
		return "<%f,%f,%f,%f>" % (x.x, x.y, x.z, x.w)
	elif isinstance(x, simulation.ObjectState):
		return "<xyz=%s vel=%s quat=%s avel=%s>" % (to_str(x.xyz), to_str(x.vel), to_str(x.quat), to_str(x.avel))
	raise NotImplementedError

class State:
	def __init__(self, position=origin, velocity=origin, rotation=identity_quat, angular_velocity=origin):
		self.position = position
		self.velocity = velocity
		self.rotation = rotation
		self.angular_velocity = angular_velocity

	def to_ObjectState(self):
		state = simulation.ObjectState()
		state.xyz.x, state.xyz.y, state.xyz.z = self.position
		state.vel.x, state.vel.y, state.vel.z = self.velocity
		state.quat.x, state.quat.y, state.quat.z, state.quat.w = self.rotation
		state.avel.x, state.avel.y, state.avel.z = self.angular_velocity
		return state

	def from_ObjectState(self, state):
		self.position = np.array([state.xyz.x, state.xyz.y, state.xyz.z])
		self.velocity = np.array([state.vel.x, state.vel.y, state.vel.z])
		self.rotation = np.array([state.quat.x, state.quat.y, state.quat.z, state.quat.w])
		self.angular_velocity = np.array([state.avel.x, state.avel.y, state.avel.z])

class Plane:
	def __init__(self, normal, constant):
		self.normal = normal
		self.constant = constant

class Box:
	def __init__(self, extents, state, mass):
		self.extents = extents
		self.state = state
		self.mass = mass

	def sync_from_cpp(self, world):
		object_state = simulation.ObjectState()
		self.pointer.get_data(object_state)
		self.state.from_ObjectState(object_state)

	def sync_to_cpp(self, world):
		self.pointer.set_data(self.state.to_ObjectState())

class World:
	def __init__(self):
		self.physics_world_pointer = simulation.PhysicsWorld()
		self.planes = []
		self.boxes = []

	def step(self, dt, substeps):
		self.physics_world_pointer.step(dt, substeps)

	def add_plane(self, plane):
		plane.pointer = simulation.StaticPlane(
			self.physics_world_pointer,
			simulation.Vec(*plane.normal),
			plane.constant,
		)
		self.planes.append(plane)

	def add_box(self, box):
		box.pointer = simulation.Box(
			self.physics_world_pointer,
			simulation.Vec(*box.extents),
			box.state.to_ObjectState(),
			box.mass,
		)
		self.boxes.append(box)

	def serialize_setup(self):
		return {
			"planes": [
				{"normal": plane.normal, "constant": plane.constant}
				for plane in self.planes
			],
			"boxes": [
				{
					"extents": box.extents,
					"mass": box.mass,
				}
				for box in self.boxes
			],
		}

	@staticmethod
	def from_serialized_setup(setup):
		world = World()
		for plane in setup["planes"]:
			world.add_plane(Plane(plane["normal"], plane["constant"]))
		for box in setup["boxes"]:
			world.add_box(Box(
				extents=box["extents"],
				state=State(),
				mass=box["mass"],
			))
		return world

	def sync_from_cpp(self):
		for box in self.boxes:
			box.sync_from_cpp(self)

	def sync_to_cpp(self):
		for box in self.boxes:
			box.sync_to_cpp(self)

	def snapshot(self):
		# Load all the objects' data from the C++ side.
		self.sync_from_cpp()
		return {
			"boxes": [
				self.snapshot_state(box.state)
				for box in self.boxes
			],
		}

	def snapshot_state(self, state):
		return {
			"xyz": tuple(state.position),
			"vel": tuple(state.velocity),
			"quat": tuple(state.rotation),
			"avel": tuple(state.angular_velocity),
		}

	def load_snapshot(self, snapshot):
		for box, box_state_snapshot in zip(self.boxes, snapshot["boxes"]):
			self.load_snapshot_state(box, box_state_snapshot)
		# Sync the loaded data back into the C++ side.
		self.sync_to_cpp()

	def load_snapshot_state(self, box, box_state_snapshot):
		box.state.position = np.array(box_state_snapshot["xyz"])
		box.state.velocity = np.array(box_state_snapshot["vel"])
		box.state.rotation = np.array(box_state_snapshot["quat"])
		box.state.angular_velocity = np.array(box_state_snapshot["avel"])

class Trajectory:
	def __init__(self, world):
		self.world = world
		self.data = {
			"setup": world.serialize_setup(),
			"snapshots": [],
		}

	def save_snapshot(self):
		self.data["snapshots"].append(self.world.snapshot())

	def step(self, dt, substeps):
		self.world.step(dt, substeps)
		self.save_snapshot()

	@staticmethod
	def from_json(desc):
		world = World.from_serialized_setup(desc["setup"])
		traj = Trajectory(world)
		traj.data["snapshots"] = desc["snapshots"]
		return traj

if __name__ == "__main__":
	world = World()
	world.add_plane(Plane([0, 1, 0], 0))
	world.add_box(Box(
		extents=[1, 1, 1],
		state=State(
			position=[0, 10, 0],
		),
		mass=1,
	))
	world.add_box(Box(
		extents=[1, 1, 1],
		state=State(
			position=[1.1, 2, 0.12],
		),
		mass=1,
	))

	traj = Trajectory(world)
	traj.save_snapshot()
	for _ in xrange(100):
		traj.step(0.03, 60)

	with open("trajectory.json", "w") as f:
		json.dump(traj.data, f, indent=2)
		f.write("\n")

if False:
	snapshot = world.snapshot()
	print("Start:", snapshot)

	print(world.snapshot())
	world.step(0.03, 60)
	print(world.snapshot())

	world.load_snapshot(snapshot)
	print(world.snapshot())

