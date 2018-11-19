#!/usr/bin/python

import math, json
import numpy as np
import physics

class Muscle:
	MUSCLE_STRENGTH = 30.0

	def __init__(self, box1, box2, resting_length):
		self.box1 = box1
		self.box2 = box2
		self.resting_length = resting_length

	def compute_length(self):
		"""compute_length(self) -> current length of the muscle

		The muscle keeps track of the Boxes on each end, and can read their positions from their states.
		"""
		return np.linalg.norm(self.box1.state.position - np.array(self.box2.state.position))

	def compute_ddt_length(self):
		"""compute_ddt_length(self) -> rate of change of the muscle's length"""
		vector = self.compute_muscle_vector()
		return self.box1.state.velocity.dot(vector) - self.box2.state.velocity.dot(vector)

	def compute_muscle_vector(self):
		"""compute_muscle_vector(self) -> normalized vector from box1 to box2"""
		vector = self.box1.state.position - np.array(self.box2.state.position)
		vector /= np.linalg.norm(vector)
		return vector

	def apply_input(self, muscle_input):
		assert -1 <= muscle_input <= 1
		vector = self.MUSCLE_STRENGTH * self.compute_muscle_vector()
		# Positive values expand the muscle, while negative values contract it.
		self.box1.pointer.apply_force(
			physics.simulation.Vec(*(+muscle_input * vector)),
			physics.simulation.Vec(0, 0, 0),
		)
		self.box2.pointer.apply_force(
			physics.simulation.Vec(*(-muscle_input * vector)),
			physics.simulation.Vec(0, 0, 0),
		)

class RobotConfiguration:	
	def __init__(self):
		# list of physics.Box
		self.boxes = []
		# list of (physics.Box, physics.Box), corresponding to muscle pairs
		self.muscles = []
		# list of physics.Box
		self.inner_ears = []

	def add_box(self, box):
		self.boxes.append(box)
		return box

	def add_muscle(self, box1, box2):
		m = Muscle(box1, box2, 0.0)
		m.resting_length = m.compute_length()
		self.muscles.append(m)

	def add_inner_ear(self, box):
		self.inner_ears.append(box)

	def apply_policy(self, policy_vector):
		assert len(policy_vector) == len(self.muscles), "%i != %i" % (len(policy_vector), len(self.muscles))
		for muscle_input, muscle in zip(policy_vector, self.muscles):
			muscle.apply_input(muscle_input)

	def compute_sensor_data(self):
		# inner_ear_data consists of 6 floats per inner ear: 3 of velocity, 3 of angular velocity.
		# XXX: TODO: Later make this be more physically reasonable IMU information.
		#            Instead it should get acceleration plus gravity, and angular acceleration.
		inner_ear_data = np.zeros((len(self.inner_ears), 2, 3))
		for i, inner_ear_box in enumerate(self.inner_ears):
			inner_ear_data[i,0,:] = inner_ear_box.state.velocity
			inner_ear_data[i,1,:] = inner_ear_box.state.angular_velocity
		# muscle_data consists of 2 floats per muscle:
		#   the total muscle strain (+1 for doubled length, 0 for resting length, -1 for zero length, etc.)
		#   the time derivative of muscle strain in Hertz
		muscle_data = np.zeros((len(self.muscles), 2))
		for i, muscle in enumerate(self.muscles):
			muscle_data[i,0] = muscle.compute_length() / muscle.resting_length - 1.0
			muscle_data[i,1] = muscle.compute_ddt_length() / muscle.resting_length
		return np.concatenate([
			inner_ear_data.flatten(),
			muscle_data.flatten(),
		])

	def compute_utility(self):
		# Currently utility is just the mean x coordinate across our boxes.
		# FIXME: Reweight by center of mass.
		return np.mean([
			box.state.position[0] + 2 * box.state.position[1]
			for box in self.boxes
		])

	def is_on_ground(self):
		return all(
			box.state.position[1] < 0.55
			for box in self.boxes
		)

class GameEngine:
	"""GameEngine"""
	TIME_PER_STEP = 0.1
	MAX_SUBSTEPS = 10
	REWARD_PER_TIME_UPRIGHT = 0.0 #0.02

	def __init__(self, robot_config, max_time):
		self.robot_config = robot_config
		self.max_time = max_time
		self.setup()

	def setup(self):
		self.world = physics.World()
		self.world.add_plane(physics.Plane([0, 1, 0], 0))
		for box in self.robot_config.boxes:
			self.world.add_box(box)
		self.initial_state = GameState(parent_engine=self, parent_state=None, total_steps=0)

class GameState:
	def __init__(self, parent_engine, parent_state, total_steps):
		self.parent_engine = parent_engine
		self.parent_state = parent_state
		self.total_steps = total_steps
		# Capture information from parent_engine.world's current state.
		self.snapshot = parent_engine.world.snapshot()
		self.sensor_data = parent_engine.robot_config.compute_sensor_data()

	def compute_utility(self):
		self.parent_engine.world.load_snapshot(self.snapshot)
		base_utility = self.parent_engine.robot_config.compute_utility()
		return base_utility + self.total_steps * GameEngine.REWARD_PER_TIME_UPRIGHT

	def is_game_over(self):
		# Early out if the entire robot is on the ground.
		self.parent_engine.world.load_snapshot(self.snapshot)
		if self.parent_engine.robot_config.is_on_ground():
			return True
		return self.total_steps >= self.parent_engine.max_time

	def execute_policy(self, policy):
		world = self.parent_engine.world
		world.load_snapshot(self.snapshot)
		self.parent_engine.robot_config.apply_policy(policy)
		world.step(self.parent_engine.TIME_PER_STEP, self.parent_engine.MAX_SUBSTEPS)
		return GameState(self.parent_engine, self, self.total_steps + 1)

def build_demo_game_engine():
	robot = RobotConfiguration()
	boxes = []
	S = 1.2
	def new(x, y, z):
		b = physics.Box(
			extents=[0.5, 0.5, 0.5],
			state=physics.State(
				position=[x - 2 * S, 0.5 + y, z],
			),
			mass=1,
		)
		boxes.append(b)
		robot.add_box(b)
		return b
	callbacks = []
	def link(b1, b2):
		def _():
			vec = b2.state.position - np.array(b1.state.position)
			engine.world.physics_world_pointer.add_constraint(
				b1.pointer,
				b2.pointer,
				physics.simulation.Vec(*(+vec / 2.0)),
				physics.simulation.Vec(*(-vec / 2.0)),
			)
		callbacks.append(_)
	def leg(x, z):
		bottom = new(x, S * 0, z)
		middle = new(x, S * 1, z)
		top = new(x, S * 2, z)
		link(bottom, middle)
		link(middle, top)
		return top

	rear1  = leg(0*S, -S)
	rear2  = leg(0*S, +S)
	front1 = leg(3*S, -S)
	front2 = leg(3*S, +S)

	middle1 = new(0*S, 2*S, 0)
	middle2 = new(1*S, 2*S, 0)
	middle3 = new(2*S, 2*S, 0)
	middle4 = new(3*S, 2*S, 0)

	link(middle1, middle2)
	link(middle2, middle3)
	link(middle3, middle4)
	link(rear1, middle1)
	link(rear2, middle1)
	link(front1, middle4)
	link(front2, middle4)

	for i, b1 in enumerate(boxes):
		for b2 in boxes[:i]:
			robot.add_muscle(b1, b2)

#	# Add a muscle connecting each box to the box two down the line.
#	for b1, b2 in zip(boxes, boxes[2:]):
#		robot.add_muscle(b1, b2, 2.2)

	engine = GameEngine(robot, max_time=100)
	for f in callbacks:
		f()

#	# Connect each box to the next.
#	for b1, b2 in zip(boxes, boxes[1:]):

	return engine

def build_demo_game_engine_old():
	robot = RobotConfiguration()
	boxes = [
		robot.add_box(physics.Box(
			extents=[0.5, 0.5, 0.5],
			state=physics.State(
				position=[1.1 * (i - 4.5), 0.5, math.sin(i) * 0.5],
			),
			mass=1,
		))
		for i in xrange(10)
	]
	for b in boxes:
		robot.add_inner_ear(b)
	# Add a muscle connecting each box to the box two down the line.
	for b1, b2 in zip(boxes, boxes[2:]):
		robot.add_muscle(b1, b2, 2.2)

	engine = GameEngine(robot, max_time=100)

	# Connect each box to the next.
	for b1, b2 in zip(boxes, boxes[1:]):
		engine.world.physics_world_pointer.add_constraint(
			b1.pointer,
			b2.pointer,
			physics.simulation.Vec(+1.1 / 2.0, 0, 0),
			physics.simulation.Vec(-1.1 / 2.0, 0, 0),
		)

	return engine

if __name__ == "__main__":
	engine = build_demo_game_engine()
	muscle_count = len(engine.robot_config.muscles)
	sensor_dims = len(engine.initial_state.sensor_data)

	print "Muscle count:", muscle_count 
	print "Sensor dims:", sensor_dims

	traj = physics.Trajectory(engine.world)
	state = engine.initial_state
	for i in xrange(250):
		traj.save_snapshot()
		policy = np.clip(np.random.randn(muscle_count), -1, 1)
#		if i == 125:
#			state = engine.initial_state
#			engine.world.load_snapshot(engine.initial_state.snapshot)
		state = state.execute_policy(policy)

	with open("/tmp/trajectory.json", "w") as f:
		json.dump(traj.data, f)

