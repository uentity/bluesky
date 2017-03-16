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

template< class T >
BS_API_PLUGIN std::string serialize_to_str(const std::shared_ptr< T >& t) {
	typedef typename boost::archive::text_oarchive archive_t;
	std::ostringstream os;
	// ensure archive is destroyed before stream is closed
	{
		archive_t ar(os);
		serialize_fix_data< archive_t >(ar) << t;
	}
	return os.str();
}

template< class T >
BS_API_PLUGIN std::shared_ptr< T > serialize_from_str(const std::string& src) {
	typedef typename boost::archive::text_iarchive archive_t;
	std::istringstream is(src);
	std::shared_ptr< T > t;
	// ensure archive is destroyed before stream is closed
	{
		archive_t ar(is);
		serialize_fix_data< archive_t >(ar) >> t;
	}
	return t;
}

template< class T >
BS_API_PLUGIN std::shared_ptr< T > copy_via_serialize(const std::shared_ptr< T >& src) {
	return serialize_from_str< T >(serialize_to_str< T >(src));
}

/*-----------------------------------------------------------------
 * indirect string serialization -- can be used with abstract interfaces
 *----------------------------------------------------------------*/
// 2-nd param in both functions used only to pass type information
// R is a temporary type used to actually invoke serialization sunctions inside

template< class T, class R >
BS_API_PLUGIN std::string serialize_to_str_indirect(const std::shared_ptr< T >& t) {
	return serialize_to_str(std::static_pointer_cast< R, T >(t));
}

template< class T, class R >
BS_API_PLUGIN std::shared_ptr< T > serialize_from_str_indirect(const std::string& src) {
	return std::static_pointer_cast< T, R >(serialize_from_str< R >(src));
}

}  // eof blue_sky namespace

