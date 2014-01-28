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

