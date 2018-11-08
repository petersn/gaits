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

def translate_xyz(v):
	return v[2], v[0], v[1]

def translate_quat(q):
	r = mathutils.Quaternion((0, 0, 1, 1)).normalized()
	return r * q.conjugated() * r.conjugated()

def copy_world_to_blender():
	for box in world.boxes:
		box.obj.location = translate_xyz(box.state.position)
		box.obj.rotation_euler = translate_quat(mathutils.Quaternion(box.state.rotation)).to_euler()

traj.world.load_snapshot(traj.data["snapshots"][0])
copy_world_to_blender()

def frame_change(scene):
	i = bpy.context.scene.frame_current
	i = max(0, min(len(traj.data["snapshots"]) - 1, i))
	traj.world.load_snapshot(traj.data["snapshots"][i])
	copy_world_to_blender()

bpy.app.handlers.frame_change_post.append(frame_change)

