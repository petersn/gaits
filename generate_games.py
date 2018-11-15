#!/usr/bin/python

import json, logging, base64
import numpy as np
import mcts
import game

MAX_STEP_RATIO = 10

def generate_game(args):
	game_engine = game.build_demo_game_engine()

	entry = {"samples": [], "outcome": None}
	m = mcts.MCTS(game_engine.initial_state)
	all_steps = 0
	collapse = 0
	while True:
		if m.root_node.state.is_game_over():
			break
		most_visits = 0
		while most_visits < args.visits and m.root_node.all_edge_visits < args.visits * MAX_STEP_RATIO:
			all_steps += 1
			edge = m.step()
			most_visits = max(most_visits, edge.edge_visits)

		# Compute the proportion of visits each move received.
		total_visits = float(m.root_node.all_edge_visits)
		weighted_moves = {
			move: (
				m.root_node.outgoing_edges[move].edge_visits / total_visits
				if move in m.root_node.outgoing_edges
				else 0
			)
			for move in m.root_node.state.moves
		}
		if weighted_moves["m0"] == 1.0:
			collapse += 1

		# Mix the policies by their visit counts to get a training policy.
		training_policy = np.sum([
			weight * m.root_node.state.moves[move].policy
			for move, weight in weighted_moves.iteritems()
		], axis=0)

		# Store training sample.
		entry["samples"].append((
			map(float, m.root_node.state.sensor_data),
			map(float, training_policy),
			float(m.root_node.visit_weighted_edge_score()),
		))

		# Step using the most visited move (with no noise).
		selected_move = mcts.sample_by_weight(weighted_moves)
		m.play(selected_move)

	entry["outcome"] = m.root_node.state.compute_utility()

	if collapse:
		print "!"*100, "Collapses:", collapse

	return entry

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--network", metavar="PATH", default="", help="Path to the model to load.")
	parser.add_argument("--output", metavar="PATH", type=str, default=None, help="Path to write .json games to. Writes in append mode, so it won't overwrite existing games.")
	parser.add_argument("--visits", metavar="N", default=50, type=int, help="When generating moves for games perform MCTS steps until the PV move has at least N visits.")

	args = parser.parse_args()

	mcts.initialize_model(args.network)

	with open(args.output, "a") as f:
		games_generated = 0
		while True:
			entry = generate_game(args)
			s = json.dumps(entry)
			s = base64.b64encode(s.encode("bz2"))
			assert "\n" not in s
			f.write(s)
			f.write("\n")
			f.flush()
			games_generated += 1
			print "Games generated:", games_generated, "utility =", entry["outcome"]

