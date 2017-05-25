/// @file
/// @author uentity
/// @date 25.05.2017
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "decl.h"
#include <cstdint>
#include <limits>

namespace blue_sky {

/*-----------------------------------------------------------------
 * special proxy object for processing non-fitting int numbers when loading
 *----------------------------------------------------------------*/
template< class next_fix = int >
struct serialize_fix_int {
	typedef next_fix next;

	// idea: load via maximum width integer type capable to hold any integral value
	template< class Archive, class T >
	static void do_fix_load(Archive& ar, T& v) {
		typename std::conditional<std::is_signed<T>::value, std::intmax_t, std::uintmax_t >::type tmp = 0;
		ar >> tmp;
		v = static_cast<T>(tmp);
	}
};

template< class T, class next_fixer >
struct serialize_fix_applicable< T, serialize_fix_int< next_fixer > > {
	typedef std::false_type on_save;
	typedef std::is_integral<T> on_load;
};

} // eof blue_sky namespace

