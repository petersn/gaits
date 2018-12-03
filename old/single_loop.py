#!/usr/bin/python

import os, collections, itertools, logging, random, json, base64
import numpy as np
import tensorflow as tf
import physics
import model
import game

TEMPERATURE = 0.6
ADDITIVE_TEMP = 0.6

logging.basicConfig(
	format="[%(process)5d] %(message)s",
	level=logging.DEBUG,
)

initialized = False
def initialize_model(path):
	global network, sess, initialized
	assert not initialized
	sess = tf.Session()
	network = model.load_model(sess, path, "net/", build_training=True)
	initialized = True

def make_network_input(game_state, history_len=10):
	history = [game_state]
	while len(history) < history_len:
		history.append(history[-1].parent_state or history[-1])
	return np.array([h.sensor_data for h in history]).flatten()

def compute_network(game_state):
	(policy_output,), ((value_output,),) = sess.run(
		[network.policy_output, network.value_output],
		feed_dict={
			network.input_ph: [make_network_input(game_state)],
			network.is_training_ph: False,
		},
	)
	noise = np.exp(TEMPERATURE * np.random.standard_normal(policy_output.shape))
	additive_noise = ADDITIVE_TEMP * np.random.standard_normal(policy_output.shape)
	policy_output = np.clip(policy_output * noise + additive_noise, -1, 1)
	return policy_output, value_output

def run_episode(trajectory_path=None):
	engine = game.build_demo_game_engine()
	if trajectory_path != None:
		traj = physics.Trajectory(engine.world)
	state = engine.initial_state
	samples = []
	previous_utility = state.compute_utility()
	while not state.is_game_over():
		policy, baseline_value = compute_network(state)
		state = state.execute_policy(policy)
		utility = state.compute_utility()
		samples.append({
			"sensors": make_network_input(state), #.sensor_data.tolist(),
			"policy": policy.tolist(),
			"baseline_value": baseline_value,
			"reward": utility - previous_utility,
			"utility": utility, 
		})
		previous_utility = utility
		if trajectory_path != None:
			traj.save_snapshot()
	if trajectory_path != None:
		with open(trajectory_path, "w") as f:
			json.dump(traj.data, f)
			f.write("\n")
	return samples

GAMMA = 0.99

def gamma_discount(samples):
	assert isinstance(samples, list)
	reversed_rewards = [i["reward"] for i in samples][::-1]
	reversed_gamma_discounted = []
	accum = 0.0
	for reward in reversed_rewards:
		accum = reward + GAMMA * accum
		reversed_gamma_discounted.append(accum)
	gamma_discounted = reversed_gamma_discounted[::-1]
	assert len(gamma_discounted) == len(samples)
	for reward, sample in zip(gamma_discounted, samples):
		sample["reward"] = reward

def model_path(model_number):
	return args.network_format % (model_number,)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--network-format", metavar="STR", required=True, help="Format string that yields all the networks.")
	parser.add_argument("--init-random", action="store_true", help="Instead of doing a training loop just initialize model 0 randomly.")
	parser.add_argument("--total-episodes", type=int, default=10000, help="Number of episodes to run.")
	args = parser.parse_args()

	if args.init_random:
		print "===== Initializing a random model ====="
		network = model.Network(
			#[2400, 192, 192, 192, 192],
			[4480, 192, 192, 192, 192],
			[192, 120],
			[192, 32, 1],
			"net/",
		)
		print "Total parameters:", network.total_parameters
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		model.save_model(sess, network, model_path(0))
		exit()

	# Compute the first missing model
	model_number = 0
	while os.path.exists(model_path(model_number + 1)):
		print "Model %i already exists, skipping." % (model_number,)
		model_number += 1

	print "Loading model:", model_path(model_number)
	initialize_model(model_path(model_number))

	total_training_samples = 0

	utilities = []
	#itertools.count(1):
	for episode_number in xrange(1, 1 + args.total_episodes):
	#for episode_number in xrange(1, 21):
		if episode_number % 25 == 0:
#			print "Saving trajectory."
			samples = run_episode("/tmp/trajectory.json")
		else:
			samples = run_episode()
		gamma_discount(samples)
		utilities.append(samples[-1]["utility"])
		total_training_samples += len(samples)

		batch = {"features": [], "policies": [], "values": [], "loss_multipliers": []}
		for sample in samples:
			batch["features"].append(sample["sensors"])
			# Train towards what we actually ended up doing.
			batch["policies"].append(sample["policy"])
			# Train the values towards the actual gamma discounted reward.
			batch["values"].append([sample["reward"]])
			# Use the difference between the gamma discounted reward and the baseline prediction as an advantage.
			batch["loss_multipliers"].append(sample["reward"] - sample["baseline_value"])
		_, policy_loss, value_loss = sess.run(
			[network.train_step, network.policy_loss, network.value_loss],
			feed_dict={
				network.input_ph:              batch["features"],
				network.desired_policy_ph:     batch["policies"],
				network.desired_value_ph:      batch["values"],
				network.loss_multiplier_ph:    batch["loss_multipliers"],
				network.learning_rate_ph:      0.00025,
				network.policy_loss_weight_ph: 25.0,
				network.is_training_ph:        True,
			},
		)
		if episode_number % 25 == 0:
			recent_utility = np.mean(utilities[-50:])
			print "Episode: %3i (%7i samples) -- Utility: %.5f -- Policy loss: %.5f -- Value loss: %.5f" % (
				episode_number,
				total_training_samples,
				recent_utility,
				policy_loss,
				value_loss,
			)

		if episode_number % 1000 == 0:
			output_path = model_path(model_number + 1)
			model.save_model(sess, network, output_path)
			model_number += 1

