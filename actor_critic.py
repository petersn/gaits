#!/usr/bin/python

import random, logging, itertools
import numpy as np
import tensorflow as tf
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

def compute_network(features):
	(policy_output,), ((value_output,),) = sess.run(
		[network.policy_output, network.value_output],
		feed_dict={
			network.input_ph: [features],
		},
	)
	return policy_output, value_output

def sample_from_policy(policy_output):
	# Currently this just means add noise.
	noise = np.exp(TEMPERATURE * np.random.standard_normal(policy_output.shape))
	additive_noise = ADDITIVE_TEMP * np.random.standard_normal(policy_output.shape)
	policy_output = np.clip(policy_output * noise + additive_noise, -1, 1)
	return policy_output

def play_episode(
	gamma=0.95,
	episode_length=100,
	learning_rate_policy=1e-3,
	learning_rate_value=1e-3,
):
	engine = game.build_demo_game_engine()
	state = engine.initial_state
	features2 = make_network_input(state)

	while not state.is_game_over():
		features1 = features2
		policy1, value1 = compute_network(features1)
		utility1 = state.compute_utility()

		action = sample_from_policy(policy1)
		state = state.execute_policy(action)

		features2 = make_network_input(state)
		policy2, value2 = compute_network(features2)
		utility2 = state.compute_utility()
		reward = utility2 - utility1

		delta = reward + gamma * value2 - value1

		# Update the network.
		sess.run(
			network.train_step,
			feed_dict={
				network.input_ph: [features1],
				network.selected_action_ph: [action],
				network.policy_loss_weight_ph: learning_rate_policy * delta,
				network.value_loss_weight_ph: learning_rate_value * delta,
				network.learning_rate_ph: -1,
			},
		)

	return state.compute_utility()

if __name__ == "__main__":
	initialize_model("runs/run-pg2-6/model-027.nn")

	print "Running episodes."
	with open("rewards", "w") as f:
		pass
	ewma = 0.0
	for episode_number in itertools.count(1):
		r = play_episode()
		ewma *= 0.999
		ewma += r * 0.001
		print "[%5i] Reward: %8.5f  EWMA: %8.5f" % (episode_number, r, ewma)
		with open("rewards", "a") as f:
			print >>f, r

