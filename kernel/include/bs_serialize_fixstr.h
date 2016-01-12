/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_SERIALIZE_FIXSTR_YG59DP85
#define BS_SERIALIZE_FIXSTR_YG59DP85

#include "bs_serialize_decl.h"
#include "bs_misc.h"
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_same.hpp>

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
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};

} /* blue_sky */


#endif /* end of include guard: BS_SERIALIZE_FIXSTR_YG59DP85 */

