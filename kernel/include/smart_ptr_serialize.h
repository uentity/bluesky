/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef SMART_PTR_SERIALIZE_27TX7GHW
#define SMART_PTR_SERIALIZE_27TX7GHW

#include "smart_ptr.h"
#include "st_smart_ptr_serialize.h"

#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/integral_c_tag.hpp>

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/tracking.hpp>
// include shared_ptr serialization
//#include <boost/serialization/shared_ptr.hpp>

namespace boost { namespace serialization {

// turn off smart pointers tracking
template< class T, bool F >
struct tracking_level< ::blue_sky::smart_ptr< T, F > > {
	typedef mpl::integral_c_tag tag;
	typedef mpl::int_< ::boost::serialization::track_never> type;
	BOOST_STATIC_CONSTANT(int, value = type::value);
};

template< class T >
struct tracking_level< ::blue_sky::mt_ptr< T > > {
	typedef mpl::integral_c_tag tag;
	typedef mpl::int_< ::boost::serialization::track_never> type;
	BOOST_STATIC_CONSTANT(int, value = type::value);
};

/*-----------------------------------------------------------------
 * Serialization of smart_ptr to BlueSky types
 *----------------------------------------------------------------*/
// save smart pointer to BlueSky type
template< class Archive, class T >
void save(
	Archive& ar,
	const blue_sky::smart_ptr< T, true >& t,
	const unsigned int /* version */
) {
	const T* t_ptr = t.get();
	ar << make_nvp("px", t_ptr);
}

// load smart pointer to BlueSky type
template< class Archive, class T >
void load(
	Archive& ar,
	blue_sky::smart_ptr< T, true >& t,
	const unsigned int /* version */
) {
	T* r;
	ar >> make_nvp("px", r);
	t = r;
}

template< class Archive, class T >
void serialize(
	Archive& ar,
	blue_sky::smart_ptr< T, true >& t,
	const unsigned int version
){
	// The most common cause of trapping here would be serializing
	// something like shared_ptr<int>.  This occurs because int
	// is never tracked by default.  Wrap int in a trackable type
	BOOST_STATIC_ASSERT((tracking_level< T >::value != track_never));

	split_free(ar, t, version);
}

/*-----------------------------------------------------------------
 * Serialization of smart_ptr to any type
 *----------------------------------------------------------------*/
// in order to properly ser-ze smart_ptr< T, false >, we need to store
// redirect to base class (st_smart_ptr) serialization
//
template< class Archive, class T >
void serialize(
	Archive& ar,
	blue_sky::smart_ptr< T, false >& t,
	const unsigned int /* version */
){
	ar & base_object< blue_sky::smart_ptr< T, false >::base_t >(t);
}

/*-----------------------------------------------------------------
 * Serialization of mt_ptr
 *----------------------------------------------------------------*/

}} /* boost::serialization */


#endif /* end of include guard: SMART_PTR_SERIALIZE_27TX7GHW */

