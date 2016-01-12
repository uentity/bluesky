/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief BlueSky typeinfo declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_TYPEINFO_H
#define _BS_TYPEINFO_H

#include "setup_common_api.h"
#include <typeinfo>
#include <cassert>

#define BS_TYPE_INFO blue_sky::bs_type_info

#define BS_GET_TI(T) (BS_TYPE_INFO(typeid(T)))

namespace blue_sky
{
	////////////////////////////////////////////////////////////////////////////////
	// class type_info
	// Purpose: offer a first-class, comparable wrapper over std::type_info
	////////////////////////////////////////////////////////////////////////////////

	//empty class denotes "Nil" type - associated with nothing
	//includes plugs for
	class BS_API nil;

	class BS_API bs_type_info
	{
	public:
		// Constructors
		bs_type_info(); // needed for containers
		bs_type_info(const std::type_info&); // non-explicit

		// Access for the wrapped std::type_info
		const std::type_info& get() const;
		// Compatibility functions
		bool before(const bs_type_info& rhs) const;
		const char* name() const;
		//check if this instance describes nothing (Nil)
		bool is_nil() const;

	private:

		const std::type_info* pinfo_;
	};

	// Implementation

	inline bs_type_info::bs_type_info(const std::type_info& ti)
		: pinfo_(&ti)
	{ assert(pinfo_); }

	inline bool bs_type_info::before(const bs_type_info& rhs) const
	{
		assert(pinfo_);
		// type_info::before return type is int in some VC libraries
		return pinfo_->before(*rhs.pinfo_) != 0;
	}

	inline const std::type_info& bs_type_info::get() const
	{
		assert(pinfo_);
		return *pinfo_;
	}

	inline const char* bs_type_info::name() const
	{
		assert(pinfo_);
		return pinfo_->name();
	}

	// Comparison operators

	inline bool operator==(const bs_type_info& lhs, const bs_type_info& rhs)
		// type_info::operator== return type is int in some VC libraries
	{ return (lhs.get() == rhs.get()) != 0; }

	inline bool operator!=(const bs_type_info& lhs, const bs_type_info& rhs)
	{ return !(lhs == rhs); }

	inline bool operator<(const bs_type_info& lhs, const bs_type_info& rhs)
	{ return lhs.before(rhs); }

	inline bool operator<=(const bs_type_info& lhs, const bs_type_info& rhs)
	{ return (lhs < rhs || lhs == rhs); }

	inline bool operator>(const bs_type_info& lhs, const bs_type_info& rhs)
	{ return !(rhs <= lhs); }

	inline bool operator>=(const bs_type_info& lhs, const bs_type_info& rhs)
	{ return !(lhs < rhs); }
}

#endif //_BS_TYPEINFO_H
