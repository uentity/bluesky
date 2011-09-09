/// @file bs_npvec_shared.h
/// @brief traits for bs_array with reshape and other numpy_array feautures support
/// based on bs_array_shared to prevent data copying to/from Python
/// @author uentity
/// @version 
/// @date 08.09.2011
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

#ifndef BS_NPVEC_SHARED_J87TDMGT
#define BS_NPVEC_SHARED_J87TDMGT

#include "bs_array_shared.h"
#include <boost/python.hpp>
#include <boost/python/errors.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_categories.hpp>
#include <boost/range.hpp>

#include <numpy/arrayobject.h>

namespace blue_sky {

template< class T >
class BS_API bs_npvec_shared : public bs_array_shared< T > {
public:
	typedef bs_npvec_shared< T > this_t;
	typedef bs_array_shared< T > base_t;

	typedef typename base_t::container container;
	typedef bs_arrbase< T > arrbase;
	typedef this_t bs_array_base;

	typedef typename arrbase::sp_arrbase sp_arrbase;

	// inherited from bs_arrbase class
	typedef typename arrbase::value_type value_type;
	typedef typename arrbase::key_type   key_type;
	typedef typename arrbase::size_type  size_type;

	typedef typename arrbase::pointer                pointer;
	typedef typename arrbase::reference              reference;
	typedef typename arrbase::const_pointer          const_pointer;
	typedef typename arrbase::const_reference        const_reference;
	typedef typename arrbase::iterator               iterator;
	typedef typename arrbase::const_iterator         const_iterator;
	typedef typename arrbase::reverse_iterator       reverse_iterator;
	typedef typename arrbase::const_reverse_iterator const_reverse_iterator;

	using base_t::resize;
	using base_t::size;
	using base_t::data;
	using base_t::begin;
	using base_t::end;

	// ctors needed by bs_array
	bs_npvec_shared() : dims_(1, 0) {}
	bs_npvec_shared(const container& c) : base_t(c), dims_(1) {
		dims_[0] = size();
	}
	// given size & fill value
	bs_npvec_shared(size_type sz, const value_type& v = value_type()) : base_t(sz, v), dims_(1) {
		dims_[0] = size();
	}

	// ctors-aliases for init
	bs_npvec_shared(size_type ndim, const npy_intp* dims) {
		init(ndim, dims);
	}

	bs_npvec_shared(size_type ndim, const npy_intp* dims, const_pointer data) {
		init(ndim, dims, data);
	}

	template< typename in_t >
	bs_npvec_shared(in_t const& in,
		typename boost::enable_if< boost::is_class< in_t > >::type *dummy = 0)
	{
		init< in_t >(in, dummy);
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
		copy(data, data + sz, begin());
		return sz;
	}

	template< typename in_t >
	size_type init(in_t const& in,
		typename boost::enable_if< boost::is_class< in_t > >::type *dummy = 0)
	{
		npy_intp dims[] = { boost::size(in) };
		size_type sz = init(1, dims);
		if(this->size())
			std::copy(boost::begin(in), boost::end(in), begin());
		return sz;
	}

	// bs_arrbase interface
	sp_arrbase clone() const {
		return new this_t(ndim(), dims(), data());
	}

	void resize(size_type new_size, const value_type& v) {
		size_type old_size = this->size();
		if(new_size != old_size) {
			resize(new_size);
			std::fill(data() + std::min(new_size, old_size), data() + new_size, v);
		}
	}

	void swap(bs_npvec_shared& rhs) {
		base_t::swap(rhs);
		std::swap(dims_, rhs.dims_);
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
		return sizeof(T);
	}

	// shape manipulation
	void reshape(int ndim, const npy_intp *dims) {
		init(ndim, dims);
	}

	void init_inplace(const container& c) {
		// call parent implementation to switch container
		base_t::init_inplace(c);
		// assume that we have plain vector
		dims_.resize(1);
		dims_[0] = size();
	}

	// switch container and provide explicit info about dims
	void init_inplace(const container& c, size_type ndim, const npy_intp* dims) {
		// switch container
		base_t::init_inplace(c);
		// copy shape info
		dims_.resize(ndim);
		std::copy(dims, dims + ndim, dims_.begin());
		// check that size match shape
		size_type shape_sz = size_from_shape();
		if(shape_sz != size())
			resize(shape_sz);
	}

private:
	std::vector< npy_intp > dims_;

	size_type size_from_shape() const {
		size_type sz = 1;
		for(ulong i = 0; i < dims_.size(); ++i)
			sz *= dims_[i];
		return sz;
	}

	size_type resize_from_shape() {
		size_type sz = size_from_shape();
		resize(sz);
		return sz;
	}
};

} /* blue_sky */

#endif /* end of include guard: BS_NPVEC_SHARED_J87TDMGT */

