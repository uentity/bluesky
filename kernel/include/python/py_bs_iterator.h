/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _PY_BS_ITERATOR_H
#define _PY_BS_ITERATOR_H

#include "bs_common.h"
#include "bs_exception.h"
#include "py_bs_object_base.h"
#include <boost/python/object.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>

#include <list>

namespace blue_sky {
namespace python {

inline boost::python::object pass_through(boost::python::object const& o) { return o; }

template<class Klass, class KlassIter>
struct iterator_wrappers {
	static Klass next(KlassIter& o) {
		Klass* result = &o.next().get();
		if (!result) {
			PyErr_SetString(PyExc_StopIteration, "No more data.");
			boost::python::throw_error_already_set();
		}
		return *result;
	}

	static void wrap(const char* python_name) {
		//using namespace boost::python;
		boost::python::class_<KlassIter>(python_name, boost::python::no_init)
			.def("next", next)
			.def("__iter__", pass_through);
	}
};

template< class T, class wrapper_type >
	class BS_API py_iterator : public std::list<wrapper_type> {
public:
	typedef smart_ptr< T > sp_T;
	typedef std::list< sp_T > sp_T_list;
	typedef typename sp_T_list::const_iterator sp_T_iter;

  explicit py_iterator(const sp_T_iter &titer) : iter(titer) {}
	explicit py_iterator() {}

	const py_iterator next() const {
		sp_T_iter tmp_iter(iter);
		return py_iterator(++tmp_iter);//+1);
	}

	wrapper_type get() const {
		if (iter != sp_T_iter()) {
			return wrapper_type(*iter);
		}
		else
			throw bs_exception("blue_sky::python::iterator","is not initialised!");
	}

protected:
	sp_T_iter iter;
};

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _PY_BS_ITERATOR_H
