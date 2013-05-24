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

	template< class Archive >
	static void do_fix_save(Archive& ar, const wsvector& v) {
		ar << (const std::size_t&)v.size();
		for(typename wsvector::const_iterator i = v.begin(), end = v.end(); i != end; ++i)
			ar << (const std::string&)wstr2str(*i);
	}

	template< class Archive >
	static void do_fix_load(Archive& ar, wsvector& v) {
		std::size_t sz;
		ar >> sz;
		v.resize(sz);
		std::string s;
		for(typename wsvector::iterator i = v.begin(), end = v.end(); i != end; ++i) {
			ar >> s;
			*i = str2wstr(s);
		}
	}
};

// allow fixer to be applied to std::wstring
template< class next_fixer >
struct serialize_fix_applicable< std::wstring, serialize_fix_wstring< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};

// and to vector of wstrings
template< class next_fixer >
struct serialize_fix_applicable< std::vector< std::wstring >, serialize_fix_wstring< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};

} /* blue_sky */


#endif /* end of include guard: BS_SERIALIZE_FIXSTR_YG59DP85 */

