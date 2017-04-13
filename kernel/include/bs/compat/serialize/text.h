/// @file
/// @author uentity
/// @date 28.10.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../../common.h"
#include "fix.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <sstream>

namespace blue_sky {

/*-----------------------------------------------------------------
 * helper functions that serialize any BS type to/from str
 *----------------------------------------------------------------*/
// R is a type that is actually serialized
// there should exist static conversion between T and R

template< class T, class R = T >
std::string serialize_to_str(const std::shared_ptr< T >& t) {
	typedef typename boost::archive::text_oarchive archive_t;
	std::ostringstream os;
	// ensure archive is destroyed before stream is closed
	{
		std::shared_ptr< R > tmp(std::static_pointer_cast< R >(t));
		archive_t ar(os);
		serialize_fix_data< archive_t >(ar) << tmp;
	}
	return os.str();
}

// specializaton for pure BS type pointers (via shared_from_thos)
template< class T, class R = T >
std::string serialize_to_str(const T* src) {
	return serialize_to_str(src->template bs_shared_this< const R >());
}

template< class T, class R = T >
std::shared_ptr< T > serialize_from_str(const std::string& src) {
	typedef typename boost::archive::text_iarchive archive_t;
	std::istringstream is(src);
	std::shared_ptr< R> t;
	// ensure archive is destroyed before stream is closed
	{
		archive_t ar(is);
		serialize_fix_data< archive_t >(ar) >> t;
	}
	return std::static_pointer_cast< T >(t);
}

template< class T >
std::shared_ptr< T > copy_via_serialize(const std::shared_ptr< T >& src) {
	return serialize_from_str< T >(serialize_to_str< T >(src));
}

template< class T >
std::shared_ptr< T > copy_via_serialize(const T* src) {
	return serialize_from_str< T >(serialize_to_str(src));
}

}  // eof blue_sky namespace

