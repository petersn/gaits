// Simulation

#include "simulation.h"

PhysicsWorld::PhysicsWorld() {
	broadphase = new btDbvtBroadphase();
	collisionConfiguration = new btDefaultCollisionConfiguration();
	dispatcher = new btCollisionDispatcher(collisionConfiguration);
	solver = new btSequentialImpulseConstraintSolver;
	dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
	dynamicsWorld->setGravity(btVector3{0, -9.8, 0});
}

void PhysicsWorld::step(float dt, int substeps) {
	dynamicsWorld->stepSimulation(dt, substeps);
}

btVector3 PhysicsObject::get_position() {
	btTransform trans;
	motionState->getWorldTransform(trans);
	return trans.getOrigin();
}

void PhysicsObject::set_data(ObjectState* state) {
	btTransform trans;
	trans.setOrigin(state->xyz.as_btVector3());
	trans.setRotation(state->quat.as_btQuaternion());
	rigidBody->setWorldTransform(trans);
	motionState->setWorldTransform(trans);
	rigidBody->setLinearVelocity(state->vel.as_btVector3());
	rigidBody->setAngularVelocity(state->avel.as_btVector3());
	// Wake up the object, in case it fell asleep from inactivity.
	rigidBody->btCollisionObject::setActivationState(ACTIVE_TAG);
}

void PhysicsObject::get_data(ObjectState* state_out) {
	btTransform trans;
	motionState->getWorldTransform(trans);
	state_out->xyz.from_btVector3(trans.getOrigin());
	state_out->vel.from_btVector3(rigidBody->getLinearVelocity());
	state_out->quat.from_btQuaternion(trans.getRotation());
	state_out->avel.from_btVector3(rigidBody->getAngularVelocity());
}

void PhysicsObject::apply_force(btVector3 force, btVector3 world_space_offset) {
	rigidBody->applyForce(force, world_space_offset);
	rigidBody->btCollisionObject::setActivationState(ACTIVE_TAG);
}

// XXX: DRY with above performance?
void PhysicsObject::apply_force(Vec* force, Vec* world_space_offset) {
	rigidBody->applyForce(force->as_btVector3(), world_space_offset->as_btVector3());
	rigidBody->btCollisionObject::setActivationState(ACTIVE_TAG);
}

Box::Box(PhysicsWorld* parent, Vec* dimensions, ObjectState* state, float mass) {
	this->parent = parent;
	shape = new btBoxShape(btVector3{dimensions->x, dimensions->y, dimensions->z});
	motionState = new btDefaultMotionState(btTransform(state->quat.as_btQuaternion(), state->xyz.as_btVector3()));
	btVector3 inertia(0, 0, 0);
	shape->calculateLocalInertia(mass, inertia);
	btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(mass, motionState, shape, inertia);
	rigidBody = new btRigidBody(rigidBodyCI);
	// Add ourselves into the parent's simulation.
	parent->dynamicsWorld->addRigidBody(rigidBody);
	parent->objects.push_back(this);
}

StaticPlane::StaticPlane(PhysicsWorld* parent, Vec* normal, float constant) {
	this->parent = parent;

	shape = new btStaticPlaneShape(normal->as_btVector3(), constant);
	motionState = new btDefaultMotionState(btTransform(
		btQuaternion::getIdentity(),
		btVector3{0, 0, 0}
	));
	btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(0, motionState, shape, btVector3{0, 0, 0});
	rigidBody = new btRigidBody(rigidBodyCI);
	// Add ourselves into the parent's simulation.
	parent->dynamicsWorld->addRigidBody(rigidBody);
	parent->objects.push_back(this);
}

void PhysicsWorld::add_constraint(PhysicsObject* obj1, PhysicsObject* obj2, Vec* offset1, Vec* offset2) {
	btTransform t1, t2;
	t1.setRotation(btQuaternion::getIdentity());
	t2.setRotation(btQuaternion::getIdentity());
	t1.setOrigin(offset1->as_btVector3());
	t2.setOrigin(offset2->as_btVector3());
	btGeneric6DofSpringConstraint* spring = new btGeneric6DofSpringConstraint(
		*obj1->rigidBody, *obj2->rigidBody,
		t1, t2,
		true
	);
	dynamicsWorld->addConstraint(spring);
}

// Fast robot implementation.

btVector3 MuscleEntry::compute_offset() {
	return obj1->get_position() - obj2->get_position();
}

float MuscleEntry::compute_length() {
	return compute_offset().norm();
}

float MuscleEntry::compute_ddt_length() {
	btVector3 v = compute_offset().normalized();
	return obj1->rigidBody->getLinearVelocity().dot(v) - obj2->rigidBody->getLinearVelocity().dot(v);
}

void MuscleEntry::apply_input(float muscle_input, float muscle_strength) {
	assert(-1 <= muscle_input);
	assert(muscle_input <= +1);
	btVector3 v = muscle_strength * compute_offset().normalized();
	obj1->apply_force((+muscle_input) * v, btVector3{0, 0, 0});
	obj2->apply_force((-muscle_input) * v, btVector3{0, 0, 0});
}

void Robot::add_component(PhysicsObject* obj) {
	components.push_back(obj);
}

void Robot::add_muscle(PhysicsObject* obj1, PhysicsObject* obj2) {
	muscles.push_back(MuscleEntry{
		obj1,
		obj2,
		0.0, // We're about to fill in this incorrect entry.
	});
	muscles.back().resting_length = muscles.back().compute_length();
}

void Robot::add_inner_ear(PhysicsObject* obj) {
	inner_ears.push_back(obj);
}

void Robot::apply_policy(int policy_vector_length, uint64_t _policy_vector) {
	float* policy_vector = reinterpret_cast<float*>(_policy_vector);
	assert(policy_vector_length == muscles.size());
	for (int i = 0; i < muscles.size(); i++)
		muscles[i].apply_input(policy_vector[i], muscle_strength);
}

void Robot::compute_sensor_data(int sensor_data_buffer_length, uint64_t _sensor_data_buffer) {
	float* sensor_data_buffer = reinterpret_cast<float*>(_sensor_data_buffer);
	float* final_position = sensor_data_buffer + sensor_data_buffer_length;

	ObjectState state;
	// Write inner ear data.
	for (PhysicsObject* obj : inner_ears) {
		obj->get_data(&state);
		sensor_data_buffer[0] = state.vel.x;
		sensor_data_buffer[1] = state.vel.y;
		sensor_data_buffer[2] = state.vel.z;
		sensor_data_buffer[3] = state.avel.x;
		sensor_data_buffer[4] = state.avel.y;
		sensor_data_buffer[5] = state.avel.z;
		sensor_data_buffer += 6;
	}
	// Write muscle data.
	for (MuscleEntry& muscle : muscles) {
		sensor_data_buffer[0] = muscle.compute_length() / muscle.resting_length - 1.0;
		sensor_data_buffer[1] = muscle.compute_ddt_length() / muscle.resting_length;
		sensor_data_buffer += 2;
	}

	assert(sensor_data_buffer == final_position);
}

float Robot::compute_utility() {
	float utility;
	for (PhysicsObject* obj : components) {
		btVector3 pos = obj->get_position();
		utility += pos.getX() + 2 * pos.getY();
	}
	utility /= components.size();
}

bool Robot::is_on_ground() {
	for (PhysicsObject* obj : components) {
		if (obj->get_position().getY() >= 0.55)
			return false;
	}
	return true;
}

