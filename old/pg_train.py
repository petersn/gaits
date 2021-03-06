#!/usr/bin/python

import os, json, random
import numpy as np
import tensorflow as tf
import model

GAMMA = 0.94

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
		sample["sensors"] = np.array(sample["sensors"])
		sample["policy"] = np.array(sample["policy"])

def load_entries(paths):
	all_samples = []
	for path in paths:
		with open(path) as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				line = line.decode("base64").decode("bz2")
				desc = json.loads(line)
				assert isinstance(desc, list)
				# Apply gamma discounting in place.
				gamma_discount(desc)
				all_samples.extend(desc)
#				for sensor_data, training_policy, score in desc["samples"]:
#					all_samples.append({
#						"features": np.array(sensor_data),
#						"policy": np.array(training_policy),
#						# This mixture here implements a form of knowledge distillation.
#						"value": np.array([0.5 * score + 0.5 * desc["outcome"]]),
#					})
	random.shuffle(all_samples)
	return all_samples

def make_minibatch(entries, size):
	batch = {"features": [], "policies": [], "values": [], "loss_multipliers": []}
	for _ in xrange(size):
		entry = random.choice(entries)
		batch["features"].append(entry["sensors"])
		batch["policies"].append(entry["policy"])
#		batch["values"].append(entry["value"])
		batch["loss_multipliers"].append(entry["reward"])
	return batch

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--games", metavar="PATH", required=True, nargs="+", help="Path to games files.")
	parser.add_argument("--old-path", metavar="PATH", help="Path for input network.")
	parser.add_argument("--new-path", metavar="PATH", required=True, help="Path for output network.")
	parser.add_argument("--steps", metavar="COUNT", type=int, default=1000, help="Training steps.")
	parser.add_argument("--minibatch-size", metavar="COUNT", type=int, default=512, help="Minibatch size.")
	parser.add_argument("--learning-rate", metavar="LR", type=float, default=0.001, help="Learning rate.")
	parser.add_argument("--policy-loss-weight", metavar="X", type=float, default=1000.0, help="Scale that's applied to the policy loss.")
	args = parser.parse_args()
	print "Arguments:", args

	# Make all choices deterministically.
	random.seed(123456789)

	all_samples = load_entries(args.games)
	print "Found %i samples." % (len(all_samples),)

	# Compute the mean reward.
	all_rewards = [sample["reward"] for sample in all_samples]
	mean_reward = np.mean(all_rewards)
	print "Reward mean:", mean_reward, " -- variance:", np.var(all_rewards)

	for sample in all_samples:
#		sample["reward"] = max(0, sample["reward"])
		sample["reward"] -= mean_reward #/ 3.0
		if sample["reward"] < 0:
			sample["reward"] /= 3.0

	sess = tf.InteractiveSession()

	if args.old_path != None:
		print "Loading old model."
		network = model.load_model(sess, args.old_path, "net/", build_training=True)
	else:
		print "WARNING: Not loading a previous model!"
		network = model.Network(
			[240, 128, 128],
			[128, 120],
			[128, 1],
			"net/",
		)
		network.build_training()
		sess.run(tf.global_variables_initializer())

	if args.steps != 0:
		in_sample_val_set = make_minibatch(all_samples, 256)

	# Begin training.
	for step_number in xrange(args.steps):
		if step_number % 5 == 0:
			policy_loss = network.run_on_samples(network.policy_loss.eval, in_sample_val_set, policy_loss_weight=args.policy_loss_weight)
#			value_loss  = network.run_on_samples(network.value_loss.eval, in_sample_val_set, policy_loss_weight=args.policy_loss_weight)
#			loss = network.get_loss(in_sample_val_set)
			print "Step: %4i -- policy loss: %.6f" % (
				step_number,
				policy_loss,
			)
		minibatch = make_minibatch(all_samples, args.minibatch_size)
		network.train(minibatch, learning_rate=args.learning_rate, policy_loss_weight=args.policy_loss_weight)

	# Write out the trained model.
	model.save_model(sess, network, args.new_path)

