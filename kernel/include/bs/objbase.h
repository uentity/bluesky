/// @file
/// @author uentity
/// @date 05.08.2016
/// @brief Base class for all BlueSky objects
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "atoms.h"
#include "error.h"
#include "type_descriptor.h"

#include <caf/actor.hpp>
#include <caf/typed_actor.hpp>

// shortcut for quick declaration of shared ptr to BS object
#define BS_SP(T) std::shared_ptr<T>

NAMESPACE_BEGIN(blue_sky)

/// @brief Base class for all BS objects
class BS_API objbase : public std::enable_shared_from_this<objbase> {
	friend class tree::link_impl;
	friend class tree::link_actor;
	friend class tree::node;
	friend class atomizer;

public:
	/// function that performs any action on this object passed as argument
	using modificator_f = std::function< error(sp_obj) >;
	using closed_modificator_f = std::function< error() >;

	/// Interface of object actor, you can only send messages matching it
	using actor_type = caf::typed_actor<
		// runs modificator in message queue of this object
		caf::replies_to<a_apply, closed_modificator_f>::with<error::box>
	>;

	// return objects's typed actor handle
	auto actor() const {
		return caf::actor_cast<actor_type>(raw_actor());
	}

	template<typename T>
	static auto actor(const T& obj) {
		return caf::actor_cast<typename T::actor_type>(obj.raw_actor());
	}

	/// default ctor that accepts custom ID string
	/// if ID is empty it will be auto-generated
	objbase(std::string custom_oid = "");
	/// default copy ctor
	objbase(const objbase&);
	/// default move ctor
	objbase(objbase&&) = default;
	// virtual destructor
	virtual ~objbase();

	/// default move assignment is fine
	objbase& operator=(objbase&&) = default;
	/// copy-assignment - will make a copy with different ID
	objbase& operator=(const objbase& rhs);

	auto swap(objbase& rhs) -> void;

	/// `shared_from_this()` casted to derived type
	template<typename Derived>
	auto bs_shared_this() const {
		return std::static_pointer_cast<const Derived, const objbase>(
			this->shared_from_this()
		);
	}

	template<typename Derived>
	auto bs_shared_this() {
		return std::static_pointer_cast<Derived, objbase>(
			this->shared_from_this()
		);
	}

	///////////////////////////////////////////////////////////////////////////////
	//  Core API that can't be shadowed in derived types
	//  [NOTE] using `virtual ... final` trick ('virtuality' will be optimized away by compiler)
	//
	/// obtain type ID: for C++ types typeid is type_descriptor.name
	virtual auto type_id() const -> std::string final;
	/// obtain object's ID
	virtual auto id() const -> std::string final;

	/// check if object is actually a tree::node
	virtual auto is_node() const -> bool final;

	/// access inode (if exists)
	virtual auto info() const -> result_or_err<tree::inode> final;

	///////////////////////////////////////////////////////////////////////////////
	//  Type-related API that derived types must provide
	//
	/// type_descriptor of objbase class
	static auto bs_type() -> const type_descriptor&;
	/// derived types must override this and return correct `type_descriptor`
	virtual auto bs_resolve_type() const -> const type_descriptor&;

	/// runs modificator in message queue of this object
	auto apply(modificator_f m) -> error;
	auto apply(launch_async_t, modificator_f m) -> void;

	template<typename F>
	auto make_closed_modificator(F&& f) -> closed_modificator_f {
		return [f = std::forward<F>(f), self = shared_from_this()]() mutable -> error {
			return f(std::move(self));
		};
	}

protected:
	/// string ID storage
	std::string id_;

	/// ctor that can delay engine start
	objbase(std::string custom_oid, bool start_actor);

	/// return object's raw (dynamic-typed) actor handle
	auto raw_actor() const -> const caf::actor&;

	/// maually start internal actor (if not started already)
	auto start_engine() -> bool;

private:
	/// flag indicating that this object is actually a tree::node
	bool is_node_;
	/// pointer to associated inode
	std::weak_ptr<tree::inode> inode_;
	/// object's internal actor
	caf::actor actor_;

	/// dedicated ctor that sets `is_node` flag
	objbase(bool is_node, std::string custom_oid = "");
};

// alias
using sp_obj  = std::shared_ptr<objbase>;
using sp_cobj = std::shared_ptr<const objbase>;

using obj_ptr  = object_ptr<objbase>;
using cobj_ptr = object_ptr<const objbase>;

NAMESPACE_END(blue_sky)
