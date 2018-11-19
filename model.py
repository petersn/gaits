#!/usr/bin/python

import sys, json
import numpy as np
import tensorflow as tf

NONLINEARITY = tf.nn.relu

class Network:
	total_parameters = 0

	def __init__(self, tower_sizes, policy_sizes, value_sizes, scope_name):
		self.tower_sizes = tower_sizes
		self.policy_sizes = policy_sizes
		self.value_sizes = value_sizes
		self.scope_name = scope_name
		with tf.variable_scope(scope_name):
			self.build_graph()

	def build_fully_connected(self, flow, sizes):
		for i, (old, new) in enumerate(zip(sizes, sizes[1:])):
			W = self.new_weight_variable([old, new])
			b = self.new_bias_variable([new])
			flow = tf.matmul(flow, W) + b
			if i < len(sizes) - 2:
				flow = NONLINEARITY(flow)
		return flow

	def build_graph(self):
		# Construct input/output placeholders.
		self.input_ph = tf.placeholder(tf.float32, shape=[None, self.tower_sizes[0]], name="input_placeholder")
		self.desired_policy_ph = tf.placeholder(tf.float32,	shape=[None, self.policy_sizes[-1]], name="desired_policy_placeholder")
#		self.desired_value_ph = tf.placeholder(tf.float32, shape=[None, self.value_sizes[-1]], name="desired_value_placeholder")
		self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="learning_rate")
		self.policy_loss_weight_ph = tf.placeholder(tf.float32, shape=[], name="policy_loss_weight")
		self.loss_multiplier_ph = tf.placeholder(tf.float32, shape=[None], name="loss_multiplier")
		self.is_training_ph = tf.placeholder(tf.bool, shape=[], name="is_training")

		# Begin constructing the data flow.
		self.parameters = []
		tower_features = NONLINEARITY(self.build_fully_connected(self.input_ph, self.tower_sizes))
		# Apply a tanh because the policy consists of muscle control values that lie in [-1, 1]
		self.policy_output = tf.nn.tanh(
			self.build_fully_connected(tower_features, self.policy_sizes),
		)
		# The value output is unbounded in value, and gets no non-linearity.
		self.value_output = self.build_fully_connected(tower_features, self.value_sizes)

	def build_training(self):
		# Make head losses.
		self.policy_loss = self.policy_loss_weight_ph * tf.reduce_mean(
			self.loss_multiplier_ph * tf.reduce_mean(
				tf.square(self.desired_policy_ph - self.policy_output),
				axis=1,
			),
		)
#		self.value_loss = tf.reduce_mean(tf.square(self.desired_value_ph - self.value_output))
		# Make regularization loss.
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
		reg_variables = tf.trainable_variables(scope=self.scope_name)
		self.regularization_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
		# Loss is the sum of these three.
		# We throw in an aribtrary weight on the policy loss, to scale it relative to the value loss.
#		self.loss = self.policy_loss + self.value_loss + self.regularization_term
		self.loss = self.policy_loss + self.regularization_term

		# Associate batch normalization with training.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope_name)
		with tf.control_dependencies(update_ops):
			self.train_step = tf.train.MomentumOptimizer(
				learning_rate=self.learning_rate_ph, momentum=0.9).minimize(self.loss)

	def new_weight_variable(self, shape):
		self.total_parameters += np.product(shape)
		stddev = 0.5 * (2.0 / np.product(shape[:-1]))**0.5
		var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
		self.parameters.append(var)
		return var

	def new_bias_variable(self, shape):
		self.total_parameters += np.product(shape)
		var = tf.Variable(tf.constant(0.0, shape=shape))
		self.parameters.append(var)
		return var

	def train(self, samples, learning_rate, policy_loss_weight):
		self.run_on_samples(
			self.train_step.run,
			samples,
			learning_rate=learning_rate,
			policy_loss_weight=policy_loss_weight,
			is_training=True,
		)

	def get_loss(self, samples):
		return self.run_on_samples(self.loss.eval, samples)

#	def get_accuracy(self, samples):
#		results = self.run_on_samples(self.final_output.eval, samples).reshape((-1, 64 * 64))
#		#results = results.reshape((-1, 64 * 8 * 8))
#		results = np.argmax(results, axis=-1)
#		assert results.shape == (len(samples["features"]),)
#		correct = 0
#		for move, result in zip(samples["moves"], results):
#			lhs = np.argmax(move.reshape((64 * 64,)))
#			#assert lhs.shape == result.shape == (2,)
#			correct += lhs == result #np.all(lhs == result)
#		return correct / float(len(samples["features"]))

	def run_on_samples(self, f, samples, learning_rate=0.01, policy_loss_weight=1.0, is_training=False):
		return f(feed_dict={
			self.input_ph:              samples["features"],
			self.desired_policy_ph:     samples["policies"],
#			self.desired_value_ph:      samples["values"],
			self.loss_multiplier_ph:    samples["loss_multipliers"],
			self.learning_rate_ph:      learning_rate,
			self.policy_loss_weight_ph: policy_loss_weight,
			self.is_training_ph:        is_training,
		})

# XXX: This is horrifically ugly.
# TODO: Once I have a second change it to not do this horrible graph scraping that breaks if you have other things going on.
def get_batch_norm_vars(net):
	return [
		i for i in tf.global_variables(scope=net.scope_name)
		if "batch_normalization" in i.name and ("moving_mean:0" in i.name or "moving_variance:0" in i.name)
	]

def save_model(sess, net, path):
	x_conv_weights = [sess.run(var) for var in net.parameters]
	x_bn_params = [sess.run(i) for i in get_batch_norm_vars(net)]
	with open(path, "w") as f:
		# First write hyperparameter meta-data.
		json.dump({"tower": net.tower_sizes, "policy": net.policy_sizes, "value": net.value_sizes}, f)
		f.write("\n")
		np.save(f, [x_conv_weights, x_bn_params])
	print "\x1b[35mSaved model to:\x1b[0m", path

# XXX: Still horrifically fragile wrt batch norm variables due to the above horrible graph scraping stuff.
def load_model(sess, path, scope_name, build_training=False):
	with open(path) as f:
		metadata = json.loads(f.readline())
		x_conv_weights, x_bn_params = np.load(f)
	net = Network(
		metadata["tower"],
		metadata["policy"],
		metadata["value"],
		scope_name=scope_name,
	)
	if build_training:
		net.build_training()
		sess.run(tf.global_variables_initializer())
	assert len(net.parameters) == len(x_conv_weights), "Parameter count mismatch!"
	operations = []
	for var, value in zip(net.parameters, x_conv_weights):
		operations.append(var.assign(value))
	bn_vars = get_batch_norm_vars(net)
	assert len(bn_vars) == len(x_bn_params), "Bad batch normalization parameter count!"
	for var, value in zip(bn_vars, x_bn_params):
		operations.append(var.assign(value))
	sess.run(operations)
	return net

if __name__ == "__main__":
#	net = Network(
#		[76, 32, 32],
#		[32, 8],
#		[32, 1],
#		"net/",
#	)
	net = Network(
		[240, 128, 128],
		[128, 120],
		[128, 1],
		"net/",
	)
	print get_batch_norm_vars(net)
	print net.total_parameters

#	if True:
#		sess = tf.Session()
#		sess.run(tf.global_variables_initializer())
#		save_model(sess, net, "/tmp/main.nn")

