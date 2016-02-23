/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_SERIALIZE_FIXREAL_TYBZECMB
#define BS_SERIALIZE_FIXREAL_TYBZECMB

#include "bs_serialize_decl.h"

#include <limits>
#include <boost/type_traits/is_floating_point.hpp>

namespace blue_sky {

/*-----------------------------------------------------------------
 * special proxy object for processing invalid real numbers before saving
 *----------------------------------------------------------------*/
template< class next_fix = int >
struct serialize_fix_real {
	typedef next_fix next;

	// fix floating point types
	template< class Archive, class T >
	static void do_fix_save(Archive& ar, const T& v) {
		T r = v;
		typedef std::numeric_limits< T > nl;
#ifdef UNIX
		if(std::isnan(v))
			r = 0;
		if(std::isinf(v))
			r = nl::max();
#else
#include <float.h>
		if(_isnan(v))
			r = 0;
		if(!_finite(v))
			r = nl::max();
#endif
		if(std::abs(v) < nl::min())
			r = 0;
		if(v > nl::max())
			r = nl::max();
		if(v < -nl::max())
			r = -nl::max();
		ar << r;
	}
};

template< class T, class next_fixer >
struct serialize_fix_applicable< T, serialize_fix_real< next_fixer > > {
	typedef boost::is_floating_point< T > on_save;
	typedef boost::false_type on_load;
};

} /* blue_sky */

#endif /* end of include guard: BS_SERIALIZE_FIXREAL_TYBZECMB */

