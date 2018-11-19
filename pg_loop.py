#!/usr/bin/python

import os, subprocess, time

def run(*l):
	print "Executing:", " ".join(l)
	subprocess.check_call(l)

directory = "run-pg-3"
P = lambda path: os.path.join(directory, path)
model_path = lambda i: P("model-%03i.nn" % (i,))

# Make the first model.
if not os.path.exists(model_path(0)):
	print "\n==== Creating initial model ====\n"
	run("python", "pg_train.py", "--games", "/dev/null", "--steps", "0", "--new-path", model_path(0))

current_model = 0
while os.path.exists(model_path(current_model + 1)):
	print "Skipping", model_path(current_model), "as its successor already exists."
	current_model += 1

while True:
	start_time = time.time()
	print "===== Running model:", current_model
	run(
		"python", "policy_gradients.py",
			"--network", model_path(current_model),
			"--output", "/tmp/samples.json",
			"--episodes", "80",
	)
	run(
		"python", "pg_train.py",
			"--games", "/tmp/samples.json",
			"--old-path", model_path(current_model),
			"--new-path", model_path(current_model + 1),
			"--steps", "50",
	)
	elapsed_time = time.time() - start_time
	print "Time elapsed for loop:", elapsed_time
	current_model += 1

