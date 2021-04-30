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
#include "tree/node.h"
#include "detail/object_ptr.h"

#include <caf/actor.hpp>
#include <caf/typed_actor.hpp>
#include <caf/group.hpp>

#include <mutex>

#define BS_SP(T) std::shared_ptr<T>

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  Base class of all BS objects
 *-----------------------------------------------------------------------------*/
class BS_API objbase : public std::enable_shared_from_this<objbase> {
public:
	/// Interface of object actor, you can only send messages matching it
	using actor_type = caf::typed_actor<
		// get home group
		caf::replies_to<a_home>::with<caf::group>,
		// runs transaction in message queue of this object
		caf::replies_to<a_apply, obj_transaction>::with<tr_result::box>
	>;

	/// default ctor that accepts custom ID string
	/// if ID is empty it will be auto-generated
	objbase(std::string custom_oid = {});

	/// polymorphic support
	virtual ~objbase();

	/// copy will have the same ID as source
	objbase(const objbase&);
	auto operator=(const objbase& rhs) -> objbase&;
	/// move is actually `default`, nothing fancy
	objbase(objbase&&);
	auto operator=(objbase&&) -> objbase&;

	auto swap(objbase& rhs) -> void;

	/// type_descriptor of objbase class
	static auto bs_type() -> const type_descriptor&;

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

	/// return objects's typed actor handle
	auto actor() {
		return caf::actor_cast<actor_type>(raw_actor());
	}

	template<typename T>
	static auto actor(const T& obj) {
		return caf::actor_cast<typename T::actor_type>(obj.raw_actor());
	}

	///////////////////////////////////////////////////////////////////////////////
	//  Core API that can't be shadowed in derived types
	//  [NOTE] using `virtual ... final` trick ('virtuality' will be optimized away by compiler)
	//
	/// obtain type ID: for C++ types typeid is type_descriptor.name
	virtual auto type_id() const -> std::string final;
	/// obtain object's ID
	virtual auto id() const -> std::string final;

	/// access inode (if exists)
	virtual auto info() const -> result_or_err<tree::inode> final;

	/// get object's home group
	virtual auto home() const -> caf::group final;

	/// get objects's home group ID (empty for invalid / not started home)
	virtual auto home_id() const -> std::string final;

	/// runs modificator in message queue of this object
	virtual auto apply(obj_transaction tr) -> tr_result final;
	virtual auto apply(launch_async_t, obj_transaction tr) -> void final;

	/// sends empty transaction to trigger `data modified` signal
	virtual auto touch(tr_result tres = {}) -> void final;

	///////////////////////////////////////////////////////////////////////////////
	//  Customization points for derived types
	//
	/// derived types must override this and return correct `type_descriptor`
	virtual auto bs_resolve_type() const -> const type_descriptor&;

	/// node service - default impl returns nil node
	virtual auto data_node() const -> tree::node;

protected:
	std::string id_;

	/// return object's raw (dynamic-typed) actor handle
	auto raw_actor() -> const caf::actor&;

private:
	friend class ::cereal::access;
	friend class atomizer;
	friend class tree::link_impl;

	/// pointer to associated inode
	std::weak_ptr<tree::inode> inode_;
	/// object's internal actor
	caf::actor actor_;
	/// internal home group ID = object *unique* ID
	uuid hid_;
	std::once_flag einit_flag_;
};
// alias
using sp_obj  = std::shared_ptr<objbase>;
using sp_cobj = std::shared_ptr<const objbase>;
using obj_ptr  = object_ptr<objbase>;
using cobj_ptr = object_ptr<const objbase>;

/*-----------------------------------------------------------------------------
 *  Base class for objects that can contain nested subobjects
 *-----------------------------------------------------------------------------*/
class BS_API objnode : public objbase {
public:
	/// default ctor will start empty internal node
	objnode(std::string custom_oid = {});

	/// install external node into this objects
	objnode(tree::node N, std::string custom_oid = {});

	/// copy ctor makes deep clone of contained node
	objnode(const objnode& rhs);
	auto operator=(const objnode& rhs) -> objnode&;

	objnode(objnode&& rhs) = default;
	auto operator=(objnode&& rhs) -> objnode& = default;

	auto swap(objnode&) -> void;

	/// returns internal node
	auto data_node() const -> tree::node override;

protected:
	/// bundeled node
	tree::node node_;

	BS_TYPE_DECL
	friend class atomizer;
};
using sp_objnode = std::shared_ptr<objnode>;
using sp_cobjnode = std::shared_ptr<const objnode>;

NAMESPACE_END(blue_sky)
