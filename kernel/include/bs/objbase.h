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
#include "error.h"
#include "type_descriptor.h"

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

	// [TODO] remove these outdated methods
	// register this instance in kernel instances list
	int bs_register_this() const;
	// remove this instance from kernel instances list
	int bs_free_this() const;

protected:
	/// string ID storage
	std::string id_;

private:
	/// flag indicating that this object is actually a tree::node
	bool is_node_;
	/// pointer to associated inode
	std::weak_ptr<tree::inode> inode_;

	/// dedicated ctor that sets `is_node` flag
	objbase(bool is_node, std::string custom_oid = "");
};

// alias
using sp_obj  = std::shared_ptr<objbase>;
using sp_cobj = std::shared_ptr<const objbase>;

using obj_ptr  = object_ptr<objbase>;
using cobj_ptr = object_ptr<const objbase>;

NAMESPACE_END(blue_sky)
