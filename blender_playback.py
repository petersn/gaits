#!/usr/bin/python

import bpy, mathutils
import sys, json
sys.path.append(".")

import physics

# Delete the initial objects.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete() 

with open("trajectory.json") as f:
	desc = json.load(f)

traj = physics.Trajectory.from_json(desc)
world = traj.world

for box in world.boxes:
	bpy.ops.mesh.primitive_cube_add()
	box.obj = bpy.context.scene.objects.active
	# Set the object's dimensions.
	box.obj.scale = box.extents

def copy_world_to_blender():
	for box in world.boxes:
		box.obj.location = box.state.position
		box.obj.rotation_euler = mathutils.Quaternion(box.state.rotation).to_euler()

traj.world.load_snapshot(traj.data["snapshots"][0])
copy_world_to_blender()

def frame_change(scene):
	i = bpy.context.scene.frame_current
	i = max(0, min(len(traj.data["snapshots"]) - 1, i))
	print("Change:", i)
	traj.world.load_snapshot(traj.data["snapshots"][i])
	copy_world_to_blender()

bpy.app.handlers.frame_change_post.append(frame_change)

