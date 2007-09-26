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

/*!
 * \file bs_conversion.h
 * \brief Helper class to test conversion, equality, etc between two types in compile-time
 * Based on Loki Conversion class with some extensions
 *
 * \author uentity
 */

#ifndef _BS_CONVERSION_H
#define _BS_CONVERSION_H

#include "boost/type_traits/remove_const.hpp"

namespace blue_sky {
namespace bs_private {

template< class T, class U >
struct conversion_helper {
	typedef char Small;
	struct Big { Small dummy[2]; };
	static Small test(U*);
	static Big test(...);
};

}

template< class T, class U >
class conversion {
private:
	typedef bs_private::conversion_helper< T, U > H;
	typedef typename H::Small H_Small;

	typedef typename boost::remove_const< T >::type unconst_T;
	typedef typename boost::remove_const< U >::type unconst_U;
	typedef bs_private::conversion_helper< unconst_T, unconst_U > H_unconst;
	typedef typename H_unconst::Small H_Small_unconst;

	enum { 
		//! guard for detecting incomplete types
		use_with_complete_types = (sizeof(T) == sizeof(U))
	};

public:
	enum { exists = (sizeof(H::test(static_cast< T* >(0))) == sizeof(H_Small)) };
	enum { exists_uc = (sizeof(H_unconst::test(static_cast< unconst_T* >(0))) == sizeof(H_Small_unconst)) };
	enum { exists2way = exists && conversion< U, T >::exists };
	enum { exists1of2way = exists || conversion< U, T >::exists };
	enum { exists2way_uc = exists_uc && conversion< U, T >::exists_uc };
	enum { exists1of2way_uc = exists_uc || conversion< U, T >::exists_uc };
	enum { same_type = false };
};

template < class T >
class conversion<T, T>
{
public:
	enum { 
		exists = 1,
		exists2way = 1,
		exists1of2way = 1,
		same_type = 1,
		exists_uc = 1,
		exists2way_uc = 1,
		exists1of2way_uc = 1
	};
};

template < class T >
class conversion< void, T >
{
public:
	enum { 
		exists = 0,
		exists2way = 0,
		exists1of2way = 0,
		same_type = 0,
		exists_uc = 0,
		exists2way_uc = 0,
		exists1of2way_uc = 0
	};
};

template < class T >
class conversion< T, void >
{
public:
	enum { 
		exists = 0,
		exists2way = 0,
		exists1of2way = 0,
		same_type = 0,
		exists_uc = 0,
		exists2way_uc = 0,
		exists1of2way_uc = 0
	};
};

//! you can convert void to void
template< >
class conversion< void, void >
{
public:
	enum {
		exists = 1,
		exists2way = 1,
		exists1of2way = 1,
		same_type = 1,
		exists_uc = 1,
		exists2way_uc = 1,
		exists1of2way_uc = 1
	};
};
}

#define BS_CONVERSION(T, R) (blue_sky::conversion< T, R >::exists)
#define BS_CONVERSION_UC(T, R) (blue_sky::conversion< T, R >::exists_uc)
#define BS_ANY_WAY_CONVERSION(T, R) (blue_sky::conversion< T, R >::exists1of2way)
#define BS_ANY_WAY_CONVERSION_UC(T, R) (blue_sky::conversion< T, R >::exists1of2way_uc)

#endif
