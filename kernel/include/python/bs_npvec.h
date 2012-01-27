/// @file bs_npvec.h
/// @brief vector traits for bs_array with reshape and other numpy_array feautures support
/// @author uentity
/// @version 1.0
/// @date 07.09.2011
/// @copyright This file is part of BlueSky
///            
///            BlueSky is free software; you can redistribute it and/or
///            modify it under the terms of the GNU Lesser General Public License
///            as published by the Free Software Foundation; either version 3
///            of the License, or (at your option) any later version.
///            
///            BlueSky is distributed in the hope that it will be useful,
///            but WITHOUT ANY WARRANTY; without even the implied warranty of
///            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///            GNU General Public License for more details.
///            
///            You should have received a copy of the GNU General Public License
///            along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#ifndef BS_NPVEC_IC9OQ7JE
#define BS_NPVEC_IC9OQ7JE

#include "bs_vecbase.h"

#include <boost/python.hpp>
#include <boost/python/errors.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_categories.hpp>
#include <boost/range.hpp>

#include <numpy/arrayobject.h>

namespace blue_sky {
namespace detail {

template< class base_t >
class BS_API bs_npvec_impl : public base_t {
public:
	typedef typename base_t::value_type value_type;
	typedef typename base_t::size_type  size_type;

	typedef bs_npvec_impl this_t;
	typedef typename base_t::container container;
	typedef typename base_t::arrbase arrbase;

	typedef typename arrbase::sp_arrbase sp_arrbase;
	typedef typename arrbase::const_pointer const_pointer;

	using base_t::size;
	using base_t::data;
	using base_t::begin;
	using base_t::end;

	// ctors needed by bs_array
	bs_npvec_impl() : dims_(1, 0) {}
	bs_npvec_impl(const container& c) : base_t(c), dims_(1) {
		dims_[0] = size();
	}
	// given size & fill value
	bs_npvec_impl(size_type sz, const value_type& v = value_type()) : base_t(sz, v), dims_(1) {
		dims_[0] = size();
	}
	// std copy stor is fine

	// model numpy_array
	// just store dimensions info

	size_type init(size_type ndim, const npy_intp* dims) {
		if(!ndim) return 0;
		dims_.resize(ndim);
		std::copy(dims, dims + ndim, dims_.begin());
		// resize internal buffer
		return resize_from_shape();
	}

	size_type init(size_type ndim, const npy_intp* dims, const_pointer data) {
		size_type sz = init(ndim, dims);
		if(this->size())
			std::copy(data, data + sz, begin());
		return sz;
	}

	template< typename in_t >
	size_type init(in_t const& in,
		typename boost::enable_if< boost::is_class< in_t > >::type* = 0)
	{
		npy_intp dims[] = { boost::size(in) };
		size_type sz = init(1, dims);
		if(this->size())
			std::copy(boost::begin(in), boost::end(in), begin());
		return sz;
	}

	void swap(bs_npvec_impl& rhs) {
		base_t::swap(rhs);
		std::swap(dims_, rhs.dims_);
	}

	void resize(size_type new_size) {
		resize_to_shape(new_size);
	}

	void resize(size_type new_size, const value_type& v) {
		size_type old_size = this->size();
		if(resize_to_shape(new_size))
			std::fill(data() + std::min(new_size, old_size), data() + new_size, v);
	}

	size_type ndim() const {
		return dims_.size();
	}

	npy_intp* dims() {
		return &dims_[0];
	}

	const npy_intp* dims() const {
		return &dims_[0];
	}

	npy_intp dim(npy_intp i) const {
		return dims_[i];
	}

	npy_intp itemsize() const {
		return sizeof(value_type);
	}

	// shape manipulation
	void reshape(int ndim, const npy_intp *dims) {
		init(ndim, dims);
	}

protected:
	std::vector< npy_intp > dims_;

	size_type size_from_shape() const {
		size_type sz = 1;
		for(ulong i = 0; i < dims_.size(); ++i)
			sz *= dims_[i];
		return sz;
	}

	size_type resize_from_shape() {
		size_type sz = size_from_shape();
		if(sz != size())
			base_t::resize(sz);
		return size();
	}

	// if resize happens, discard dims info and make array vector-like
	bool resize_to_shape(size_type new_size) {
		if(new_size != size()) {
			base_t::resize(new_size);
			dims_.resize(1);
			dims_[0] = size();
			return true;
		}
		return false;
	}
};

} // eof namespace detail

template< class T >
class BS_API bs_npvec : public detail::bs_npvec_impl< bs_vecbase_impl< T, std::vector< T > > > {
public:
	typedef detail::bs_npvec_impl< bs_vecbase_impl< T, std::vector< T > > > base_t;

	typedef std::vector< T > container;
	typedef bs_arrbase< T > arrbase;
	typedef bs_npvec bs_array_base;
	typedef typename arrbase::sp_arrbase sp_arrbase;

	// inherited from bs_arrbase class
	typedef typename arrbase::value_type    value_type;
	typedef typename arrbase::size_type     size_type;
	typedef typename arrbase::const_pointer const_pointer;

	using base_t::init;

	// ctors needed by bs_array
	bs_npvec() {}
	bs_npvec(const container& c) : base_t(c) {}
	// given size & fill value
	bs_npvec(size_type sz, const value_type& v = value_type()) : base_t(sz, v) {}

	// ctors-aliases for init
	bs_npvec(size_type ndim, const npy_intp* dims) {
		init(ndim, dims);
	}

	bs_npvec(size_type ndim, const npy_intp* dims, const_pointer data) {
		init(ndim, dims, data);
	}

	template< typename in_t >
	bs_npvec(in_t const& in,
		typename boost::enable_if< boost::is_class< in_t > >::type *dummy = 0)
	{
		init(in, dummy);
	}

	sp_arrbase clone() const {
		// copy ctor will make deep copy
		return new bs_npvec(*this);
	}

	void swap(bs_npvec& rhs) {
		base_t::swap(rhs);
	}
};

} /* blue_sky */

#endif /* end of include guard: BS_NPVEC_IC9OQ7JE */

