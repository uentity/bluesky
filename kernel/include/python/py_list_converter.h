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
#ifndef PY_LIST_CONVERTER_JP85ZM43
#define PY_LIST_CONVERTER_JP85ZM43

#include "py_bs_converter.h"
#include <boost/python/list.hpp>

namespace blue_sky { namespace python {
namespace bp = boost::python;

// strat_id: 0 - allcoate vector first, then assign elements
//               fast, but requires appropriate list_t ctor (use fo vectors)
//           1 - grow list_t using push_back, only copy ctor needed (vector and list)
//           2 - grow list_t using insert (use for sets)
template< class list_t, int strat_id = 0 >
struct list_traits {
	typedef list_t type;

	typedef typename list_t::size_type size_t;
	typedef typename list_t::value_type value_t;

	template< int strat_id_, class = void >
	struct create_strat {
		static void create_type(void* mem_chunk, boost::python::object& obj) {
			// get length
			size_t sz = bp::len(obj);
			// create c++ object
			list_t* res = new(mem_chunk) list_t;
			// fill it from Python list
			for(size_t i = 0; i < sz; ++i)
				res->push_back(bp::extract< value_t >(obj[i]));
		}
	};

	template< class unused >
	struct create_strat< 0, unused > {
		static void create_type(void* mem_chunk, boost::python::object& obj) {
			// get length
			size_t sz = bp::len(obj);
			// create c++ object
			list_t* res = new(mem_chunk) list_t(sz);
			// fill it from Python list
			size_t i = 0;
			for(typename type::iterator pv = res->begin(), end = res->end(); pv != end; ++pv, ++i)
				*pv = bp::extract< value_t >(obj[i]);
		}
	};

	template< class unused >
	struct create_strat< 2, unused > {
		static void create_type(void* mem_chunk, boost::python::object& obj) {
			// get length
			size_t sz = bp::len(obj);
			// create c++ object
			list_t* res = new(mem_chunk) list_t;
			// fill it from Python list
			for(size_t i = 0; i < sz; ++i)
				res->insert(bp::extract< value_t >(obj[i]));
		}
	};

	static void create_type(void* mem_chunk, boost::python::object& obj) {
		create_strat< strat_id >::create_type(mem_chunk, obj);
	}

	static bool is_convertible(PyObject* py_obj) {
		if( !PySequence_Check( py_obj ) ){
			return false;
		}

		if( !PyObject_HasAttrString( py_obj, "__len__" ) ){
			return false;
		}

		bp::object py_seq( bp::handle<>( bp::borrowed( py_obj ) ) );
		size_t sz = bp::len(py_seq);
		for(ulong i = 0; i < sz; ++i) {
			if(!bp::extract< value_t >(py_seq[i]).check())
				return false;
		}
		return true;
	}

	static PyObject* to_python(type const& v) {
		// Make bp::object
		bp::list py_l;
		for(typename type::const_iterator pv = v.begin(), end = v.end(); pv != end; ++pv)
			py_l.append(*pv);
		// export it to Python
		return bp::incref(py_l.ptr());
	}
};

}} // eof blue_sky::python

#endif /* end of include guard: PY_LIST_CONVERTER_JP85ZM43 */

