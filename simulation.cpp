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

void PhysicsObject::set_data(ObjectState* state) {
	btTransform trans;
	trans.setOrigin(state->xyz.as_btVector3());
	trans.setRotation(state->quat.as_btQuaternion());
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


/*
btSliderConstraint* bt_constraint =
    new btSliderConstraint(
        *ref_bodyA,
        fromB,true);

bt_constraint->setLowerLinLimit(-10.0f);
bt_constraint->setUpperLinLimit(10.0f);

bt_constraint->setLowerAngLimit(0.0f);
bt_constraint->setUpperAngLimit(0.0f);
*/

void PhysicsWorld::add_constraint(PhysicsObject* obj1, PhysicsObject* obj2) {
	btTransform t1, t2;
	t1.setRotation(btQuaternion::getIdentity());
	t2.setRotation(btQuaternion::getIdentity());
	t1.setOrigin({0, -1.5, 0});
	t2.setOrigin({0, 1.5, 0});
	btGeneric6DofSpringConstraint* spring = new btGeneric6DofSpringConstraint(
		*obj1->rigidBody, *obj2->rigidBody,
		t1, t2,
		true
	);
	dynamicsWorld->addConstraint(spring);
}

