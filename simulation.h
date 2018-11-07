// Simulation

#ifndef PHYS_SIMULATION_H
#define PHYS_SIMULATION_H

#include <btBulletDynamicsCommon.h>
#include <list>

struct Vec {
	float x, y, z, w;

	Vec() : x(0), y(0), z(0), w(0) { }
	Vec(float _x, float _y, float _z) : x(_x), y(_y), z(_z), w(0) { }
	Vec(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) { }

	btVector3 as_btVector3() {
		return btVector3(x, y, z);
	}

	btQuaternion as_btQuaternion() {
		return btQuaternion(x, y, z, w);
	}

	void from_btVector3(btVector3 v) {
		x = v[0]; y = v[1]; z = v[2]; w = 0;
	}

	void from_btQuaternion(btQuaternion q) {
		x = q[0]; y = q[1]; z = q[2]; w = q[3];
	}
};

struct ObjectState {
	Vec xyz{0, 0, 0};
	Vec vel{0, 0, 0};
	Vec quat{0, 0, 0, 1};
	Vec avel{0, 0, 0};
};

struct PhysicsObject;

struct PhysicsWorld {
	btBroadphaseInterface* broadphase;
	btDefaultCollisionConfiguration* collisionConfiguration;
	btCollisionDispatcher* dispatcher;
	btSequentialImpulseConstraintSolver* solver;
	btDiscreteDynamicsWorld* dynamicsWorld;
	std::list<PhysicsObject*> objects;

	PhysicsWorld();
	void step(float dt, int substeps);
};

struct PhysicsObject {
	PhysicsWorld* parent;
	btCollisionShape* shape;
	btDefaultMotionState* motionState;
	btRigidBody* rigidBody;

	void set_data(ObjectState* state);
	void get_data(ObjectState* state_out);
	void apply_force(Vec* force, Vec* world_space_offset);
};

struct Box : public PhysicsObject {
	Box(PhysicsWorld* parent, Vec* dimensions, ObjectState* state, float mass);
};

struct StaticPlane : public PhysicsObject {
	StaticPlane(PhysicsWorld* parent, Vec* normal, float constant);
};

#endif

