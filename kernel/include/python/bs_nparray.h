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

#include "bs_arrbase.h"
#include <pyublas/numpy.hpp>
#include <boost/python/errors.hpp>

namespace blue_sky {

template< class T >
class BS_API bs_nparray : public bs_arrbase_impl< T, pyublas::numpy_array< T > > {
public:
	typedef bs_nparray< T > this_t;
	typedef pyublas::numpy_array< T > numpy_array_t;
	typedef bs_arrbase_impl< T, pyublas::numpy_array< T > > base_t;

	// traits for bs_array
	typedef numpy_array_t container;
	typedef bs_arrbase< T > arrbase;
	typedef this_t bs_array_base;

	typedef typename arrbase::sp_arrbase sp_arrbase;
	typedef typename arrbase::size_type size_type;
	typedef typename arrbase::value_type value_type;
	typedef typename arrbase::pointer pointer;

	// ctors needed by bs_array
	bs_nparray() {}
	bs_nparray(const container& c) : base_t(c) {}
	// std copy stor is fine

	// constructors via init
	bs_nparray(size_type n)
		: base_t(numpy_array_t(n))
	{}

	bs_nparray(int ndim_, const npy_intp* dims_)
		: base_t(numpy_array_t(ndim_, dims_))
	{}

	bs_nparray(size_type n, const value_type& v)
		: base_t(numpy_array_t(n, v))
	{}

	bs_nparray(const boost::python::handle<> &obj)
		: base_t(numpy_array_t(obj))
	{}

	// assume borrowed object, i.e. increment it's refcounter first
	bs_nparray(PyObject* obj)
		: base_t(numpy_array_t(
			boost::python::handle<>(boost::python::borrowed(obj))
		))
	{}

	bs_nparray(pointer data, size_type n) {
		npy_intp sz[] = { n };
		this_t(base_t(numpy_array_t(1, sz, data))).swap(*this);
	}

	sp_arrbase clone() const {
		return new bs_nparray(this->copy());
	}

	void swap(this_t& rhs) {
		base_t::swap(rhs);
	}

	PyObject* to_python() const {
		return base_t::to_python().release();
	}

	// numpy::array implementation just create new array
	// so handle resize using std numpy C iface
	void resize(size_type new_size) {
		// test if no array was created yet
		if(this->handle().get() == Py_None)
			numpy_array_t::resize(new_size);
		if(new_size == this->size()) return;

		// native resize
		npy_intp new_dims[] = { new_size };
		PyArray_Dims d = { new_dims, 1};
		try {
			boost::python::handle<> new_array = boost::python::handle<>(
				PyArray_Resize((PyArrayObject*)this->handle().get(), &d, 1, NPY_ANYORDER)
			);
			if(new_array.get() && new_array.get() != Py_None && new_array.get() != this->handle().get())
				this_t(new_array).swap(*this);
		}
		catch(const boost::python::error_already_set& e){
			// if resize fails - do nothing
			PyErr_Print();
		}
	}

	void resize(size_type new_size, value_type init) {
		size_type old_size = 0;
		if(this->handle().get())
			old_size = this->size();

		resize(new_size);
		pointer new_data = this->data();
		std::fill(new_data + std::min(old_size, new_size), new_data + new_size, init);
	}

};

} 	// eof blue_sky namespace

#endif /* end of include guard: BS_ARRAY_NUMPY_5NAYJGRI */

