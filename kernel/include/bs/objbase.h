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
//#include "messaging.h"
//#include "objbase_macro.h"

namespace blue_sky {
/*!
	\class objbase
	\ingroup object_base
	\brief This is a base class for all objects.
*/
class BS_API objbase {
	friend class kernel;
	//friend class combase;
	friend class bs_inode;
	//friend class bs_link;

	BS_COMMON_DECL(objbase)

public:
	//! swap function needed to provide assignment ability
	void swap(objbase& rhs);

	// virtual destructor
	virtual ~objbase();

	// signals list
	//BLUE_SKY_SIGNALS_DECL_BEGIN(bs_messaging)
	//	//on_unlock,
	//	on_delete,
	//BLUE_SKY_SIGNALS_DECL_END

	//! type_descriptor of objbase class
	static type_descriptor bs_type();

	/*!
	\brief Type descriptor resolver for derived class.
	This method should be overridden by childs.
	\return reference to const type_descriptor
	*/
	virtual type_descriptor bs_resolve_type() const = 0;

	// register this instance in kernel instances list
	int bs_register_this() const;
	// remove this instance from kernel instances list
	int bs_free_this() const;

	// default object deletion method - executes 'delete this'
	virtual void dispose() const;

	/*!
	\brief Access to corresponding inode object
	*/
	const blue_sky::bs_inode* inode() const;

protected:
	// associated inode
	const blue_sky::bs_inode* inode_;

	//! Necessary declarations.
	//BS_COMMON_DECL(objbase);
	//BS_LOCK_THIS_DECL(objbase);
};

}	//namespace blue_sky

