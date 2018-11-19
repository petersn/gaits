#!/usr/bin/python

import collections, logging, random, json, base64
import numpy as np
import tensorflow as tf
import physics
import model
import game

TEMPERATURE = 0.2
ADDITIVE_TEMP = 0.2

logging.basicConfig(
	format="[%(process)5d] %(message)s",
	level=logging.DEBUG,
)

initialized = False
def initialize_model(path):
	global network, sess, initialized
	assert not initialized
	sess = tf.Session()
	network = model.load_model(sess, path, "net/")
	initialized = True

def compute_policy(game_state):
	(policy_output,), ((value_output,),) = sess.run(
		[network.policy_output, network.value_output],
		feed_dict={
			network.input_ph: [game_state.sensor_data],
			network.is_training_ph: False,
		},
	)
	noise = np.exp(TEMPERATURE * np.random.standard_normal(policy_output.shape))
	additive_noise = ADDITIVE_TEMP * np.random.standard_normal(policy_output.shape)
	policy_output = np.clip(policy_output * noise + additive_noise, -1, 1)
	return policy_output

def run_episode():
	engine = game.build_demo_game_engine()
	if args.do_trajectory:
		traj = physics.Trajectory(engine.world)
	state = engine.initial_state
	samples = []
	previous_utility = state.compute_utility()
	while not state.is_game_over():
		policy = compute_policy(state)
		state = state.execute_policy(policy)
		utility = state.compute_utility()
		samples.append({
			"sensors": state.sensor_data.tolist(),
			"policy": policy.tolist(),
			"reward": utility - previous_utility,
		})
		previous_utility = utility
		if args.do_trajectory:
			traj.save_snapshot()
	if args.do_trajectory:
		with open("/tmp/trajectory.json", "w") as f:
			json.dump(traj.data, f)
			f.write("\n")
	return samples

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--network", metavar="PATH", default="", help="Path to the model to load.")
	parser.add_argument("--output", metavar="PATH", type=str, default=None, help="Path to write .json games to. Writes in append mode, so it won't overwrite existing games.")
	parser.add_argument("--do-plot", action="store_true", help="Produce a plot.")
	parser.add_argument("--do-trajectory", action="store_true", help="Save the last generated episode to /tmp/trajectory.json.")
	parser.add_argument("--episodes", default=40, type=int, help="Number of episodes to generate.")
	args = parser.parse_args()

	if args.do_plot:
		import matplotlib.pyplot as plt
		plt.hold(True)

	initialize_model(args.network)
	print "Running episodes."
	with open(args.output, "w") as f:
		episodes_generated = 0
		for _ in xrange(args.episodes):
			samples = run_episode()
			s = json.dumps(samples)
			s = base64.b64encode(s.encode("bz2"))
			assert "\n" not in s
			f.write(s)
			f.write("\n")
			f.flush()
			episodes_generated += 1
			if episodes_generated % 10 == 0:
				print "Episodes generated:", episodes_generated
			if args.do_plot:
				plt.plot([i["reward"] for i in samples])

	if args.do_plot:
		plt.show()

