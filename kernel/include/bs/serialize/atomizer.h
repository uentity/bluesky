/// @file
/// @author uentity
/// @date 04.06.2018
/// @brief Atomizer is used to access private fields of class to be serialized
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../setup_plugin_api.h"
#include <type_traits>
#include <cstdint>

namespace blue_sky {

/*-----------------------------------------------------------------------------
 *  make this class a friend to be able to access private variables
 *  from non-intrusive serialization code
 *-----------------------------------------------------------------------------*/
class BS_API_PLUGIN atomizer {
public:
	template<typename T>
	struct save {};

	template<typename T>
	struct load {};

	template<typename T>
	struct serialize {};

	template<typename T>
	struct save_minimal {};

	template<typename T>
	struct load_minimal {};

	template<typename T>
	struct load_and_construct {};
};

} // eof blue_sky namespace

/*-----------------------------------------------------------------------------
 *  if a specialization of `blue_sky::atomizer::[serialization fcn]::go` exists,
 *  enable corresponding specialization of `cereal::[serialization fcn]
 *  that forwards call to `blue_sky::atomizer` (which has access to private members)
 *-----------------------------------------------------------------------------*/
namespace cereal {
///////////////////////////////////////////////////////////////////////////////
//  forward versioned functions
//
// cereal::save -> blue_sky::atomizer::save::go
template<
	typename Archive, typename T, typename = decltype( ::blue_sky::atomizer::save<T>::go(
		std::declval<Archive&>(), std::declval<T const&>(), std::declval<std::uint32_t const>()
	))
>
auto save(Archive& ar, T const& t, std::uint32_t const version) -> void {
	::blue_sky::atomizer::save<T>::go(ar, t, version);
}

// cereal::load -> blue_sky::atomizer::load::go
template<
	typename Archive, typename T, typename = decltype( ::blue_sky::atomizer::load<T>::go(
		std::declval<Archive&>(), std::declval<T&>(), std::declval<std::uint32_t const>()
	))
>
auto load(Archive& ar, T& t, std::uint32_t const version) -> void {
	::blue_sky::atomizer::load<T>::go(ar, t, version);
}

// cereal::serialize -> blue_sky::atomizer::serialize::go
template<
	typename Archive, typename T, typename = decltype( ::blue_sky::atomizer::serialize<T>::go(
		std::declval<Archive&>(), std::declval<T&>(), std::declval<std::uint32_t const>()
	))
>
auto serialize(Archive& ar, T& t, std::uint32_t const version) -> void {
	::blue_sky::atomizer::serialize<T>::go(ar, t, version);
}

// cereal::save_minimal -> blue_sky::atomizer::save_minimal::go
template<
	typename Archive, typename T, typename = decltype( ::blue_sky::atomizer::save_minimal<T>::go(
		std::declval<Archive const&>(), std::declval<T const&>(), std::declval<std::uint32_t const>()
	))
>
auto save_minimal(Archive const& ar, T const& t, std::uint32_t const version) {
	return ::blue_sky::atomizer::save_minimal<T>::go(ar, t, version);
}

// cereal::load_minimal -> blue_sky::atomizer::load_minimal::go
template<
	typename Archive, typename T, typename V, typename = decltype(
		::blue_sky::atomizer::load_minimal<T>::go(
			std::declval<Archive const&>(), std::declval<T&>(), std::declval<V const&>(),
			std::declval<std::uint32_t const>()
		)
	)
>
auto load_minimal(Archive const& ar, T& t, V const& v, std::uint32_t const version) -> void {
	::blue_sky::atomizer::load_minimal<T>::go(ar, t, v, version);
}

///////////////////////////////////////////////////////////////////////////////
//  forward non-versioned functions
//
// cereal::save -> blue_sky::atomizer::save::go
template<
	typename Archive, typename T, typename = decltype( ::blue_sky::atomizer::save<T>::go(
		std::declval<Archive&>(), std::declval<T const&>()
	))
>
auto save(Archive& ar, T const& t) -> void {
	::blue_sky::atomizer::save<T>::go(ar, t);
}

// cereal::load -> blue_sky::atomizer::load::go
template<
	typename Archive, typename T, typename = decltype( ::blue_sky::atomizer::load<T>::go(
		std::declval<Archive&>(), std::declval<T&>()
	))
>
auto load(Archive& ar, T& t) -> void {
	::blue_sky::atomizer::load<T>::go(ar, t);
}

// cereal::serialize -> blue_sky::atomizer::serialize::go
template<
	typename Archive, typename T, typename = decltype( ::blue_sky::atomizer::serialize<T>::go(
		std::declval<Archive&>(), std::declval<T&>()
	))
>
auto serialize(Archive& ar, T& t) -> void {
	::blue_sky::atomizer::serialize<T>::go(ar, t);
}

// cereal::save_minimal -> blue_sky::atomizer::save_minimal::go
template<
	typename Archive, typename T, typename = decltype( ::blue_sky::atomizer::save_minimal<T>::go(
		std::declval<Archive const&>(), std::declval<T const&>()
	))
>
auto save_minimal(Archive const& ar, T const& t) {
	return ::blue_sky::atomizer::save_minimal<T>::go(ar, t);
}

// cereal::load_minimal -> blue_sky::atomizer::load_minimal::go
template<
	typename Archive, typename T, typename V, typename = decltype(
		::blue_sky::atomizer::load_minimal<T>::go(
			std::declval<Archive const&>(), std::declval<T&>(), std::declval<V const&>()
		)
	)
>
auto load_minimal(Archive const& ar, T& t, V const& v) -> void {
	::blue_sky::atomizer::load_minimal<T>::go(ar, t, v);
}

} // eof cereal namespace

