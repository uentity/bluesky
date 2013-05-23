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

	//static std::string do_fix_save(const std::wstring& v) {
	//	return wstr2str(v);
	//}

	template< class Archive >
	static void do_fix_save(Archive& ar, const std::wstring& v) {
		ar << (const std::string&)wstr2str(v);
	}

	template< class Archive >
	static void do_fix_load(Archive& ar, std::wstring& v) {
		std::string s;
		ar >> s;
		v = str2wstr(s);
	}
};

template< class T, class next_fixer >
struct serialize_fix_applicable< T, serialize_fix_wstring< next_fixer > > {
	typedef boost::is_same< T, std::wstring > on_save;
	typedef on_save on_load;
	typedef std::string save_ret_t;
};

} /* blue_sky */


#endif /* end of include guard: BS_SERIALIZE_FIXSTR_YG59DP85 */

