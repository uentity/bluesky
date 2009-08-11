// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

/*!
  \file bs_object_base.h
  \brief declaration of base object class for neo project
  \author Gagarin Alexander <gagrinav@ufanipi.ru>
  \date 2007-03-04
 */
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

	class BS_API objbase : virtual public bs_refcounter, public bs_messaging
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
		smart_ptr< blue_sky::bs_inode, true > inode() const;

	protected:
		//! Constructor for derived classes - accepts signals range
		objbase(const bs_messaging::sig_range_t&);

		//associated inode
		smart_ptr< blue_sky::bs_inode, true > inode_;

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
