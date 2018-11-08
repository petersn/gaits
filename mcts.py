#!/usr/bin/python

import numpy as np
import tensorflow as tf
import model

initialized = False

def initialize_model(path):
	global network, sess, initialized
	assert not initialized
	sess = tf.Session()
	network = model.load_model(sess, path, "net/")
	initialized = True

class MCTSEdge:
	def __init__(self, move, child_node, parent_node=None):
		self.move = move
		self.child_node = child_node
		self.parent_node = parent_node
		self.edge_visits = 0
		self.edge_total_score = 0

	def get_edge_score(self):
		return self.edge_total_score / self.edge_visits

	def adjust_score(self, new_score):
		self.edge_visits += 1
		self.edge_total_score += new_score

	def __str__(self):
		return "<%s %4.1f%% v=%i s=%.5f c=%i>" % (
			uai_interface.uai_encode_move(self.move),
			100.0 * self.parent_node.board.evaluations.posterior[self.move],
			self.edge_visits,
			self.get_edge_score(),
			len(self.child_node.outgoing_edges),
		)

class MCTSNode:
	def __init__(self, state, parent=None):
		self.state = state
		self.parent = parent
		self.all_edge_visits = 0
		self.outgoing_edges = {}
		self.graph_name_suffix = ""

	def total_action_score(self, move):
		if move in self.outgoing_edges:
			edge = self.outgoing_edges[move]
			u_score = MCTS.exploration_parameter * self.board.evaluations.posterior[move] * (1.0 + self.all_edge_visits)**0.5 / (1.0 + edge.edge_visits)
			Q_score = edge.get_edge_score() if edge.edge_visits > 0 else 0.0
		else:
			u_score = MCTS.exploration_parameter * self.board.evaluations.posterior[move] * (1.0 + self.all_edge_visits)**0.5
			Q_score = 0.0
		return Q_score + u_score

	def select_action(self):
		global_evaluator.populate(self.board)
		# If we have no legal moves then return None.
		if not self.board.evaluations.posterior:
			return
		# If the game is over then return None.
		if self.board.evaluations.game_over:
			return
#		# WARNING: Does this actually use Dirichlet noise? I don't think it does.
#		posterior = self.board.evaluations.posterior
#		if use_dirichlet_noise:
#			self.board.evaluations.populate_noisy_posterior()
#			posterior = self.board.evaluations.noisy_posterior
		return max(posterior, key=self.total_action_score)

	def graph_name(self, name_cache):
		if self not in name_cache:
			name_cache[self] = "n%i%s" % (len(name_cache), "") #self.graph_name_suffix)
		return name_cache[self]

	def make_graph(self, name_cache):
		l = []
		for edge in self.outgoing_edges.itervalues():
			l.append("%s -> %s;" % (self.graph_name(name_cache), edge.child_node.graph_name(name_cache)))
		for edge in self.outgoing_edges.itervalues():
			# Quadratic time here from worst case for deep graphs.
			l.extend(edge.child_node.make_graph(name_cache))
		return l

class TopN:
	def __init__(self, N, key):
		self.N = N
		self.key = key
		self.entries = []

	def add(self, item):
		if item not in self.entries:
			self.entries += [item]
		self.entries = sorted(self.entries, key=self.key)[-self.N:]

	def update(self, items):
		for i in items:
			self.add(i)

class MCTS:
	exploration_parameter = 1.0

	def __init__(self, root_board, use_dirichlet_noise=False):
		self.root_node = MCTSNode(root_board.copy())
		self.use_dirichlet_noise = use_dirichlet_noise

	def select_principal_variation(self, best=False):
		node = self.root_node
		edges_on_path = []
		while True:
			if best:
				if not node.outgoing_edges:
					break
				move = max(node.outgoing_edges.itervalues(), key=lambda edge: edge.edge_visits).move
			else:
				move = node.select_action(
					# Use dirichlet noise only if we are configured to, AND only at the root of the search tree.
					use_dirichlet_noise = self.use_dirichlet_noise and node == self.root_node,
				)
			if move not in node.outgoing_edges:
				break
			edge = node.outgoing_edges[move]
			edges_on_path.append(edge)
			node = edge.child_node
		return node, move, edges_on_path

	def step(self):
		def to_move_name(move):
			return "_%s" % (move,)
		# 1) Pick a child by repeatedly taking the best child.
		node, move, edges_on_path = self.select_principal_variation()
		# 2) If the move is non-null, expand once.
		if move != None:
			new_board = node.board.copy()
			try:
				new_board.move(move)
			except AssertionError, e:
				import sys
				print >>sys.stderr, node.board
				print >>sys.stderr, node.board.legal_moves()
				print >>sys.stderr, move in node.board.legal_moves()
				print >>sys.stderr, new_board
				print >>sys.stderr, new_board.legal_moves()
				print >>sys.stderr, move in new_board.legal_moves()
				print >>sys.stderr, self.root_node.outgoing_edges
				print >>sys.stderr, self.root_node.all_edge_visits
				print >>sys.stderr, self.root_node.select_action()
				print >>sys.stderr, self.root_node.board.evaluations.posterior
				print >>sys.stderr, self.root_node.board.evaluations.value
				del self.root_node.board.evaluations
				global_evaluator.populate(self.root_node.board)
				print >>sys.stderr, self.root_node.board.evaluations.posterior
				print >>sys.stderr, self.root_node.board.evaluations.value
				print >>sys.stderr, "Move:", move
				raise e
			new_node = MCTSNode(new_board, parent=node)
			new_node.graph_name_suffix = to_move_name(move)
			new_edge = node.outgoing_edges[move] = MCTSEdge(move, new_node, parent_node=node)
			edges_on_path.append(new_edge)
		else:
			# 2b) If the move is null, then we had no legal moves, and just propagate the score again.
			new_node = node
		# 3a) Evaluate the new node.
		global_evaluator.populate(new_node.board)
		# 3b) Queue up some children just for efficiency.
		for m, probability in new_node.board.evaluations.posterior.iteritems():
			if probability > NNEvaluator.PROBABILITY_THRESHOLD:
				new_board = new_node.board.copy()
				new_board.move(m)
				global_evaluator.add_to_queue(new_board)
		# Convert the expected value result into a score.
		value_score = (new_node.board.evaluations.value + 1) / 2.0
		# 4) Backup.
		inverted = False
		for edge in reversed(edges_on_path):
			# Remember that each edge corresponds to an alternating player, so we have to reverse scores.
			inverted = not inverted
			value_score = 1 - value_score
			assert inverted == (edge.parent_node.board.to_move != new_node.board.to_move)
			edge.adjust_score(value_score)
			edge.parent_node.all_edge_visits += 1
		if not edges_on_path:
			self.write_graph()
			logging.debug("WARNING no edges on path!")
		# The final value of edge encodes the very first edge out of the root.
		return edge

	def play(self, player, move, print_variation_count=True):
		assert self.root_node.board.to_move == player, "Bad play direction for MCTS!"
		if move not in self.root_node.outgoing_edges:
			if print_variation_count:
				logging.debug("Completely unexpected variation!")
			new_board = self.root_node.board.copy()
			new_board.move(move)
			self.root_node = MCTSNode(new_board)
			return
		edge = self.root_node.outgoing_edges[move]
		if print_variation_count:
			logging.debug("Traversing to variation with %i visits." % edge.edge_visits)
		self.root_node = edge.child_node
		self.root_node.parent = None

	def write_graph(self):
		name_cache = {}
		with open("/tmp/mcts.dot", "w") as f:
			f.write("digraph G {\n")
			f.write("\n".join(self.root_node.make_graph(name_cache)))
			f.write("\n}\n")
		return name_cache

if __name__ == "__main__":
	initialize_model("/tmp/main.nn")

