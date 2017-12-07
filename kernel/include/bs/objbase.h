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
#include "type_descriptor.h"
#include "objbase_macro.h"

NAMESPACE_BEGIN(blue_sky)

/// inode that stores access rights, timestampts, etc
struct BS_API inode {
	// link's owner
	std::string owner;
	std::string group;

	// flags
	bool : 1;
	bool suid : 1;
	bool sgid : 1;
	bool sticky : 1;

	// access rights
	// user (owner)
	bool : 1;
	unsigned int u : 3;
	// group
	bool : 1;
	unsigned int g : 3;
	// others
	bool : 1;
	unsigned int o : 3;
};
using inodeptr = std::unique_ptr<inode>;

/*!
	\class objbase
	\ingroup object_base
	\brief This is a base class for all objects.
*/
class BS_API objbase : public std::enable_shared_from_this< objbase > {
	friend class kernel;
	friend class tree::node;

public:
	/// default ctor that accepts custom ID string
	/// if ID is empty it will be auto-generated
	objbase(std::string custom_oid = "", inodeptr i = nullptr);
	/// default copy ctor
	objbase(const objbase&);
	/// default move ctor
	objbase(objbase&&) = default;
	// virtual destructor
	virtual ~objbase();

	/// swap function needed to provide assignment ability
	void swap(objbase& rhs);

	//! type_descriptor of objbase class
	static const type_descriptor& bs_type();
	/*!
	\brief Type descriptor resolver for derived class.
	This method should be overridden by childs.
	\return reference to const type_descriptor
	*/
	virtual const type_descriptor& bs_resolve_type() const {
		return bs_type();
	}

	// register this instance in kernel instances list
	int bs_register_this() const;
	// remove this instance from kernel instances list
	int bs_free_this() const;

	template< class Derived >
	decltype(auto) bs_shared_this() const {
		return std::static_pointer_cast< const Derived, const objbase >(this->shared_from_this());
	}

	template< class Derived >
	decltype(auto) bs_shared_this() {
		return std::static_pointer_cast< Derived, objbase >(this->shared_from_this());
	}

	/// obtain type ID: for C++ types typeid is type_descriptor.name
	virtual std::string type_id() const;
	/// obtain object's ID
	virtual std::string id() const;

	/// check if object is actually a tree::node
	/// NOTE: no RTTI, no virtual functions, just checks flag
	bool is_node() const;

	/// get/set inode
	inode info() const;
	void set_info(inodeptr i);

protected:
	/// string ID storage
	std::string id_;
	/// contains object tree-related metadata
	std::unique_ptr<inode> inode_;

private:
	/// flag indicating that this object is actually a tree::node
	bool is_node_;
	/// dedicated ctor that sets `is_node` flag
	objbase(bool is_node, std::string custom_oid = "", inodeptr i = nullptr);
};

// alias
using sp_obj = std::shared_ptr< objbase >;
using sp_cobj = std::shared_ptr< const objbase >;

NAMESPACE_END(blue_sky)

