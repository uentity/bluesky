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
		if(v < nl::min())
			r = nl::min();
		if(v > nl::max())
			r = nl::max();
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

