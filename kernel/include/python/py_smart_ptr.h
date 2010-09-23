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

#ifndef PY_SMART_PTR_FUCJ0G9W
#define PY_SMART_PTR_FUCJ0G9W

/// @brief Helper allowing to use smart_ptr< T, true > in boost::python
/// when it holds pointer to const T

#include <smart_ptr.h>
#include <boost/python/implicit.hpp>
#include <boost/python/register_ptr_to_python.hpp>

namespace boost { namespace python {

// make pointee specializations to allow boost::python deduce type pointed to by smart_ptr
template < class T >
struct pointee< blue_sky::smart_ptr< T, false > > {
	typedef T type;
};

template < class T >
struct pointee< blue_sky::smart_ptr< T, true > > {
	typedef T type;
};

template < class T >
struct pointee< blue_sky::st_smart_ptr< T > > {
	typedef T type;
};

}} // eof namespace boost::python

namespace blue_sky {

// get_pointer needed to export smart_ptr to Python
template< class T >
T* get_pointer(blue_sky::st_smart_ptr< T > const & p) {
	return p.get();
}

namespace python {
using namespace Loki;

/// @brief register smart_ptr to given object type + implicit conversion to objbase
template< class T >
void register_smart_ptr() {
	namespace bp = boost::python;
	bp::register_ptr_to_python< blue_sky::smart_ptr< T, true > >();
	//bp::implicitly_convertible< blue_sky::smart_ptr< T, true >, blue_sky::smart_ptr< objbase, true > >();
}

/// @brief Same as above, but also register converion to base_t
template< class T, class base_t >
void register_smart_ptr() {
	register_smart_ptr< T >();
	boost::python::implicitly_convertible< blue_sky::smart_ptr< T, true >, blue_sky::smart_ptr< base_t, true > >();
}

/// @brief register smart_ptr to given generic (non-BS) object type
template< class T >
void register_generic_smart_ptr() {
	boost::python::register_ptr_to_python< blue_sky::smart_ptr< T, false > >();
}

/// @brief register smart_ptr to given generic (non-BS) object type
template< class T, class base_t >
void register_generic_smart_ptr() {
	register_generic_smart_ptr< T >();
	boost::python::implicitly_convertible< blue_sky::smart_ptr< T, false >, blue_sky::smart_ptr< base_t, false > >();
}

template< class T, class base_t = void >
struct auto_reg_smart_ptr {
	enum { has_refcnt = conversion< T, bs_refcounter >::exists_uc };
	enum { has_base = !conversion< base_t, void >::same_type };

	auto_reg_smart_ptr() {
		go(Int2Type< has_refcnt >(), Int2Type< has_base >());
	}
private:
	void go(Int2Type< 0 > /* has_refcnt */, Int2Type< 0 > /* has_base */) {
		register_generic_smart_ptr< T >();
	}
	void go(Int2Type< 0 > /* has_refcnt */, Int2Type< 1 > /* has_base */) {
		register_generic_smart_ptr< T, base_t >();
	}
	void go(Int2Type< 1 > /* has_refcnt */, Int2Type< 0 > /* has_base */) {
		register_smart_ptr< T >();
	}
	void go(Int2Type< 1 > /* has_refcnt */, Int2Type< 1 > /* has_base */) {
		register_smart_ptr< T, base_t >();
	}
};

}} 	// eof namespace blue_sky::python

#endif /* end of include guard: PY_SMART_PTR_FUCJ0G9W */

