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

#ifndef BS_NPVEC_SERIALIZE_DUQHKGZW
#define BS_NPVEC_SERIALIZE_DUQHKGZW

#include "bs_npvec.h"
#include "bs_npvec_shared.h"

#include "bs_serialize_macro.h"
#include "bs_serialize_overl.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>

/*-----------------------------------------------------------------
 * serialize detail::bs_npvec_impl
 *----------------------------------------------------------------*/
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(save, blue_sky::detail::bs_npvec_impl, 1)
	typedef typename type::const_iterator citerator;
	typedef typename type::size_type size_t;

	// save array dims
	size_t ndim = t.ndim();
	ar << ndim;
	ar << boost::serialization::make_array(t.dims(), ndim);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(load, blue_sky::detail::bs_npvec_impl, 1)
	typedef typename type::iterator iterator;
	typedef typename type::size_type size_t;

	// load array dims
	size_t ndim;
	ar >> ndim;
	std::vector< npy_intp > dims(ndim);
	ar >> make_array(&dims[0], ndim);
	t.init(ndim, &dims[0]);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SERIALIZE_SPLIT_T(blue_sky::detail::bs_npvec_impl, 1)

/*-----------------------------------------------------------------
 * serialize bs_npvec & bs_npvec_shared
 *----------------------------------------------------------------*/
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(serialize, blue_sky::bs_npvec, 1)
	// just serialize impl
	typedef blue_sky::detail::bs_npvec_impl< typename type::value_type > base_t;
	boost::serialization::bs_base_object< base_t, type >(t);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(serialize, blue_sky::bs_npvec_shared, 1)
	// just serialize impl
	typedef blue_sky::detail::bs_npvec_impl< typename type::value_type > base_t;
	boost::serialization::bs_base_object< base_t, type >(t);
BLUE_SKY_CLASS_SRZ_FCN_END

#endif /* end of include guard: BS_NPVEC_SERIALIZE_DUQHKGZW */

