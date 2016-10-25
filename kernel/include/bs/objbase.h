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

namespace blue_sky {
/*!
	\class objbase
	\ingroup object_base
	\brief This is a base class for all objects.
*/
class BS_API objbase : public std::enable_shared_from_this< objbase > {
	friend class kernel;
	//friend class combase;
	friend class bs_inode;
	//friend class bs_link;

	//BS_COMMON_DECL(objbase)

public:
	/// default ctor
	objbase(); //= default;

	/// default copy ctor
	objbase(const objbase&); //= default;

	/// default move ctor
	objbase(objbase&&) = default;

	// virtual destructor
	virtual ~objbase();

	//! swap function needed to provide assignment ability
	void swap(objbase& rhs);

	// signals list
	//BLUE_SKY_SIGNALS_DECL_BEGIN(bs_messaging)
	//	//on_unlock,
	//	on_delete,
	//BLUE_SKY_SIGNALS_DECL_END

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

	// default object deletion method - executes 'delete this'
	//virtual void dispose() const;

	/*!
	\brief Access to corresponding inode object
	*/
	const blue_sky::bs_inode* inode() const;

	template< class Derived >
	decltype(auto) bs_shared_this() const {
		return std::static_pointer_cast< const Derived, const objbase >(this->shared_from_this());
	}

	template< class Derived >
	decltype(auto) bs_shared_this() {
		return std::static_pointer_cast< Derived, objbase >(this->shared_from_this());
	}

protected:
	// associated inode
	const blue_sky::bs_inode* inode_;

	//! Necessary declarations.
	//BS_COMMON_DECL(objbase);
	//BS_LOCK_THIS_DECL(objbase);
};

// alias
using sp_obj = std::shared_ptr< objbase >;

}	//namespace blue_sky

