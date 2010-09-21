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

#ifndef BS_NPARRAY_5NAYJGRI
#define BS_NPARRAY_5NAYJGRI

#include "bs_array.h"
#include <pyublas/numpy.hpp>
#include <boost/python/errors.hpp>
//#include <iostream>

namespace blue_sky {

/// @brief traits for arrays with pyublas::numpy_array container
template< class T >
struct BS_API numpy_array_traits : public bs_arrbase_impl< T, pyublas::numpy_array< T > >
{};

template< class T >
class BS_API bs_nparray : public bs_array< T, numpy_array_traits > {
public:
	typedef numpy_array_traits< T > traits_t;
	typedef bs_array< T, numpy_array_traits > bs_array_t;
	typedef bs_nparray< T > this_t;
	typedef typename traits_t::container numpy_array_t;
	typedef typename traits_t::container container;
	typedef typename bs_array_t::size_type size_type;
	typedef typename bs_array_t::value_type value_type;
	typedef typename bs_array_t::pointer pointer;

	// constructors via init
	void init(size_type n) {
		this_t(numpy_array_t(n)).swap(*this);
	}

	void init(int ndim_, const npy_intp* dims_) {
		this_t(numpy_array_t(ndim_, dims_)).swap(*this);
	}

	void init(size_type n, const value_type& v) {
		this_t(numpy_array_t(n, v)).swap(*this);
	}

	void init(const boost::python::handle<> &obj) {
		this_t(numpy_array_t(obj)).swap(*this);
	}

	void init(PyObject* obj) {
		this_t(numpy_array_t(boost::python::handle<>(
						boost::python::borrowed(obj)
		))).swap(*this);
	}

	void init(pointer data, size_type n) {
		npy_intp dims[] = { n };
		init(PyArray_SimpleNewFromData(1, dims, pyublas::get_typenum(T()), data));
	}

	void test() {
		std::cout << "test" << std::endl;
	}

	void swap(this_t& rhs) {
		bs_array_t::swap(rhs);
	}

	PyObject* to_python() const {
		// its VERY IMPORTANT to return PyObject* like that
		// instead of simply handle().get(), because in the latter case
		// array seems to be deleted immediately after function exit
		return boost::python::handle<>(this->handle()).release();
	}

	//// numpy::array implementation just create new array
	//// so handle resize using std numpy C iface
	//void resize(size_type new_size) {
	//	if(new_size == this->size()) return;
	//	npy_intp new_dims[] = { new_size };
	//	PyArray_Dims d = { new_dims, 1};
	//	try {
	//		boost::python::handle<> new_array = boost::python::handle<>(
	//			PyArray_Resize((PyArrayObject*)this->handle().get(), &d, 1, NPY_ANYORDER)
	//		);
	//		if(new_array.get() && new_array.get() != Py_None)
	//			init(new_array);
	//	}
	//	catch(...) {
	//		// if resize fails - do nothing
	//		// boost::python::handle_exception();
	//	}
	//}

	//void resize(size_type new_size, value_type init) {
	//	size_type old_size = 0;
	//	if(this->handle().get())
	//		old_size = this->size();

	//	resize(new_size);
	//	pointer new_data = this->data();
	//	std::fill(new_data + std::min(old_size, new_size), new_data + new_size, init);
	//}

protected:
	// copy construct from base class
	bs_nparray(const bs_array_t& rhs)
		: bs_array_t(rhs)
	{}

	BLUE_SKY_TYPE_DECL_T(bs_nparray);
};

// default ctor
template< class T >
bs_nparray< T >::bs_nparray(bs_type_ctor_param param)
{}

// copy ctor
template< class T >
bs_nparray< T >::bs_nparray(const bs_nparray& v)
	: bs_refcounter(), bs_array_t(v)
{}

typedef bs_nparray< int > bs_nparray_i;
typedef bs_nparray< float > bs_nparray_f;
typedef bs_nparray< double > bs_nparray_d;

} 	// eof blue_sky namespace

#endif /* end of include guard: BS_ARRAY_NUMPY_5NAYJGRI */

