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

void PhysicsObject::get_data(ObjectState* state_out) {
	btTransform trans;
	motionState->getWorldTransform(trans);
	state_out->xyz.from_btVector3(trans.getOrigin());
	state_out->vel.from_btVector3(rigidBody->getLinearVelocity());
	state_out->quat.from_btQuaternion(trans.getRotation());
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

