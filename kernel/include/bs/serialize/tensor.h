/// @file
/// @author uentity
/// @date 11.11.2018
/// @brief Eigen serialization support
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../error.h"
#include "../detail/tensor_meta.h"
#include "carray.h"

#include <cereal/types/array.hpp>

namespace cereal {
template<typename T> using TensorMeta = blue_sky::meta::tensor<T>;

/*-----------------------------------------------------------------------------
 *  Save: fits any tensor type (but not TensorBase)
 *-----------------------------------------------------------------------------*/
template <typename Archive, typename TensorT>
auto save(Archive& ar, TensorT const& t)
-> std::enable_if_t<TensorMeta<TensorT>::is_handle || TensorMeta<TensorT>::is_tensor> {
	// for text archives carray gives better output
	// but for binary it saves arary size which isn't needed for std::array
	const auto shape = t.dimensions();
	if constexpr(cereal::traits::is_text_archive<Archive>::value)
		ar( make_nvp("shape", make_carray_view(shape.data(), shape.size())) );
	else
		ar( make_nvp("shape", shape) );
	// save data
	ar( make_nvp("data", make_carray_view(t.data(), t.size())) );
}

/*-----------------------------------------------------------------------------
 *  Load
 *-----------------------------------------------------------------------------*/
// for plain tensors that can be resized
template <typename Archive, typename T>
auto load(Archive& ar, T& t) -> std::enable_if_t<TensorMeta<T>::is_mutable_tensor> {
	typename T::Dimensions shape;
	if constexpr(cereal::traits::is_text_archive<Archive>::value)
		ar( make_nvp("shape", make_carray_view(shape.data(), shape.size())) );
	else
		ar( make_nvp("shape", shape) );
	// load data
	t.resize(shape);
	ar( make_nvp("data", make_carray_view(t.data(), t.size())) );
}

// maps and refs that can't be resized
template <class Archive, class T>
auto load(Archive& ar, T& t) -> std::enable_if_t<TensorMeta<T>::is_mutable_handle> {
	// read size
	typename T::Dimensions shape;
	if constexpr(cereal::traits::is_text_archive<Archive>::value)
		ar( make_nvp("shape", make_carray_view(shape.data(), shape.size())) );
	else
		ar( make_nvp("shape", shape) );
	// maybe just check size (remove that strict shape check)?
	//if(!std::equal(shape.begin(), shape.end(), t.dimensions()))
	//	throw blue_sky::error("Shape of TensorMap or TensorRef target don't match required");
	// read data with size check
	ar(make_nvp("data",
		make_carray_view(t.data(), t.size(), [&t](std::size_t read_sz) {
			if(t.size() < read_sz)
				throw blue_sky::error("Size of TensorMap or TensorRef target is less than required");
		})
	));
}

} // eof cereal
