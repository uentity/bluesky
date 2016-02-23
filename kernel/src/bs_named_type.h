/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Declaration of parent ( named_type ) of base object class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_NAMED_TYPE_H
#define _BS_NAMED_TYPE_H

#include "setup_common_api.h"
#include "type_descriptor.h"
//replaced simple counter with boost's thread-safe atomic count
#include "boost/detail/atomic_count.hpp"
#include "generic_smart_ptr.h"

namespace blue_sky {

	/*!
		\class named_type
		\ingroup blue_sky
		\brief Parent class of all blue_sky types (particularly of objbase). Contains reference counter and name.
	 */
	class BS_API named_type
	{
	public:

		//type declaration

		/*!
		\brief Type descriptor getter.
		This method should be overridden in derived classes.
		\return reference to const type_descriptor
		*/
		BS_TYPE_DECL
		virtual const type_descriptor& resolve_descriptor() const = 0;

		 //references counting

		 /*!
			 \brief Add reference.
		 */
		void add_ref() const { 
			++refcnt_;
		}

		/*!
			\brief Delete reference.
		 */
		void del_ref() const { 
		   	if(--refcnt_ <= 0) 
			   dispose();
		}
		
		/*!
			\brief Returns references count.
		*/
		long refs() const { return refcnt_; }

		/*!
			\brief Self-destruction method - default is 'delete', assumes creating with 'new'
		*/
		virtual void dispose() const {
		   delete this;
		}

		/*!
			\brief Mutex accessor.
			\return Reference to mutex
		*/
		bs_mutex& mutex() const { return mut_; }

		/*!
			\brief Name accessor for constant object
			\return multithreaded pointer
		*/
		virtual mt_ptr< std::string > name() const {
			return mt_ptr< std::string >(&name_, name_mut_);
		}

		/*!
			\brief Name accessor for non-const object.
			\return direct simple pointer
		*/
		virtual std::string* name() { return &name_; }

		//virtual destructor & constructors

		//!	Virtual destructor.
		virtual ~named_type() {};

		//! Constructor.
		named_type() : refcnt_(0) {}
		//! Copy constructor.
		named_type(const named_type& /*l*/) : refcnt_(0) {}

	private:
		//reference counter
		mutable boost::detail::atomic_count refcnt_;

		//mutex for non-const members access
		mutable bs_mutex mut_;

		//object's name
		mutable std::string name_;
		//name guard
		mutable bs_mutex name_mut_;
	};
}

#endif		//_BS_REFCNT_H
