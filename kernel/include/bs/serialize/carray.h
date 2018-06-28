/// @file
/// @author uentity
/// @date 28.06.2018
/// @brief Serialization support for C-style arrays
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <cereal/cereal.hpp>

/*-----------------------------------------------------------------------------
 *  serialization support for C arrays
 *  [NOTE] size of array should be already serialized beforehand!
 *-----------------------------------------------------------------------------*/

namespace blue_sky {
namespace detail {

// Serialization for C arrays if BinaryData is supported and we are arithmetic
template<class Archive, class T> inline
auto serialize_carray(Archive& ar, T* array, const std::size_t size, std::true_type /* binary_supported */)
-> void {
	ar( cereal::binary_data(array, sizeof(T) * size) );
}

//! Serialization for arrays if BinaryData is not supported or we are not arithmetic
/*! @internal */
template <class Archive, class T> inline
auto serialize_carray(Archive& ar, T* array, const std::size_t size, std::false_type /* binary_supported */ )
-> void {
	for(std::size_t i = 0; i < size; ++i)
		ar(array[i]);
}

// proxy view class for C arrays
template<typename T>
struct carray_view {
	T* data_;
	std::size_t size_;

	explicit carray_view(T* data, std::size_t size) : data_(data), size_(size) {}

	// test if archive supports binary data serialization
	template<typename Archive> using binary_supported = std::integral_constant<
		bool, cereal::traits::is_output_serializable<cereal::BinaryData<T>, Archive>::value &&
		std::is_arithmetic<typename std::remove_all_extents<T>::type>::value
	>;

	// main serialization function
	template<typename Archive>
	auto serialize(Archive& ar) -> void {
		serialize_carray(ar, data_, size_, binary_supported<Archive>());
	}
};

} // eof namespace detail

template<typename Archive, typename T> inline
auto serialize_carray(Archive& ar, T* array, const std::size_t size) -> void {
	detail::serialize_carray(ar, array, size, detail::carray_view<T>::template binary_supported<Archive>());
}

template<typename T>
auto make_carray_view(T* array, std::size_t size) {
	return detail::carray_view<T>(array, size);
}

} // eof namespace blue_sky

