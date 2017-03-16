/// @file
/// @author uentity
/// @date 28.10.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "decl.h"
#include "../../detail/str_utils.h"

namespace blue_sky {

/*-----------------------------------------------------------------
 * fix for converting wstring -> string before saving
 *----------------------------------------------------------------*/
template< class next_fix = int >
struct serialize_fix_wstring {
	// next is type of next fix in chain to be applied
	typedef next_fix next;
	typedef std::vector< std::wstring > wsvector;

	template< class Archive >
	static void do_fix_save(Archive& ar, const std::wstring& v) {
		ar << (const std::string&)wstr2str(v, "utf-8");
	}

	template< class Archive >
	static void do_fix_load(Archive& ar, std::wstring& v) {
		std::string s;
		ar >> s;
		v = str2wstr(s, "utf-8");
	}
};

// allow fixer to be applied to std::wstring
template< class next_fixer >
struct serialize_fix_applicable< std::wstring, serialize_fix_wstring< next_fixer > > {
	typedef std::true_type on_save;
	typedef std::true_type on_load;
};

} /* blue_sky */

