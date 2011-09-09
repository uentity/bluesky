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
#include "bs_npvec.h"

namespace blue_sky {

template< class T >
class BS_API bs_npvec_shared : public detail::bs_npvec_impl< bs_array_shared< T > > {
public:
	typedef bs_npvec_shared this_t;
	typedef bs_array_shared< T > array_shared_t;
	typedef detail::bs_npvec_impl< bs_array_shared< T > > base_t;

	typedef typename array_shared_t::container container;
	typedef bs_arrbase< T > arrbase;
	typedef this_t bs_array_base;

	typedef typename arrbase::sp_arrbase sp_arrbase;

	// inherited from bs_arrbase class
	typedef typename arrbase::value_type    value_type;
	typedef typename arrbase::size_type     size_type;
	typedef typename arrbase::const_pointer const_pointer;

	using base_t::init;
	using base_t::size;
	using base_t::dims_;

	// ctors needed by bs_array
	bs_npvec_shared()  {}
	bs_npvec_shared(const container& c) : base_t(c) {}
	// given size & fill value
	bs_npvec_shared(size_type sz, const value_type& v = value_type()) : base_t(sz, v) {}

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
		init(in, dummy);
	}

	// bs_arrbase interface
	sp_arrbase clone() const {
		return new this_t(this->ndim(), this->dims(), this->data());
	}

	void swap(bs_npvec_shared& rhs) {
		base_t::swap(rhs);
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
		this->resize_from_shape();
	}
};

} /* blue_sky */

#endif /* end of include guard: BS_NPVEC_SHARED_J87TDMGT */

