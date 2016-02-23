/// @file
/// @author uentity
/// @date 04.03.2007
/// @brief Base class for all BlueSky objects
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_OBJECT_BASE_H
#define _BS_OBJECT_BASE_H

#include "bs_common.h"
#include "bs_refcounter.h"
#include "type_descriptor.h"
#include "bs_messaging.h"

#include "bs_object_base_macro.h"



namespace blue_sky {
/*!
 * \brief object_base ... short description ...
 * \author Gagarin Alexander <gagrinav@ufanipi.ru>
 * \date 2007-03-04
 * ... description ...
 */

/*!
	\class objbase
	\ingroup object_base
	\brief This is a base class for all objects.
*/

	class BS_API objbase : public bs_messaging
	{
		friend class kernel;
		friend class combase;
		friend class bs_inode;
		friend class bs_link;

	public:
		//typedef smart_ptr< objbase, true > sp_obj;

		//signals list
		BLUE_SKY_SIGNALS_DECL_BEGIN(bs_messaging)
			on_unlock,
			on_delete,
		BLUE_SKY_SIGNALS_DECL_END

		//! type_descriptor of objbase class
		static type_descriptor bs_type();

		/*!
		\brief Type descriptor resolver for derived class.
		This method should be overridden by childs.
		\return reference to const type_descriptor
		*/
		virtual type_descriptor bs_resolve_type() const = 0;

		//register this instance in kernel instances list
		int bs_register_this() const;
		//remove this instance from kernel instances list
		int bs_free_this() const;

		//default object deletion method - executes 'delete this'
		virtual void dispose() const;

		//virtual destructor
		virtual ~objbase();

		/*!
		\brief Access to corresponding inode object
		*/
		const blue_sky::bs_inode* inode() const;
		//smart_ptr< blue_sky::bs_inode, true > inode() const;

    template <typename class_t>
    static class_t &
    python_exporter (class_t &class__)
    {
      return class__;
    }

	protected:
		//! Constructor for derived classes - accepts signals range
		objbase(const bs_messaging::sig_range_t&);

		//associated inode
		const blue_sky::bs_inode* inode_;
		//smart_ptr< blue_sky::bs_inode, true > inode_;

		//! swap function needed to provide assignment ability
		void swap(objbase& rhs);

		//! assignamnt operator
		//objbase& operator =(const objbase& obj);

		//! Necessary declarations.
		BS_COMMON_DECL(objbase);
		BS_LOCK_THIS_DECL(objbase);
	};

}	//namespace blue_sky

#endif /* #ifndef _BS_OBJECT_BASE_H */
