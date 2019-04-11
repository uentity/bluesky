/// @file
/// @author uentity
/// @date 11.11.2018
/// @brief Eigen serialization support
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include "carray.h"
#include <Eigen/Dense>

#include <cereal/types/array.hpp>

namespace cereal {

/*-----------------------------------------------------------------------------
 *  `Array` or `Matrix`
 *-----------------------------------------------------------------------------*/
template <typename Archive, typename Derived>
auto save(Archive& ar, Eigen::PlainObjectBase<Derived> const& t) -> void {
	//using ArrT = Eigen::PlainObjectBase<Derived>;
	// write shape
	const auto shape = std::array{ t.rows(), t.cols() };
	if constexpr(cereal::traits::is_text_archive<Archive>::value)
		ar( make_nvp("shape", make_carray_view(shape.data(), shape.size())) );
	else
		ar( make_nvp("shape", shape) );
	// write data
	ar( make_nvp("data", make_carray_view(t.data(), t.size())) );
}

template <typename Archive, typename Derived>
auto load(Archive& ar, Eigen::PlainObjectBase<Derived>& t) -> void {
	//using ArrT = Eigen::PlainObjectBase<Derived>;
	// read shape
	std::array<Eigen::Index, 2> shape;
	if constexpr(cereal::traits::is_text_archive<Archive>::value)
		ar( make_nvp("shape", make_carray_view(shape.data(), shape.size())) );
	else
		ar( make_nvp("shape", shape) );
	// read data
	t.resize(shape[0], shape[1]);
	ar( make_nvp("data", make_carray_view(t.data(), t.size())) );
}

/*-----------------------------------------------------------------------------
 *  `Map`, `Ref`, `Block`, etc
 *-----------------------------------------------------------------------------*/
template <typename Archive, typename Derived, Eigen::AccessorLevels A>
auto save(Archive& ar, Eigen::MapBase<Derived, A> const& t) -> void {
	// write shape
	const auto shape = std::array{ t.rows(), t.cols() };
	if constexpr(cereal::traits::is_text_archive<Archive>::value)
		ar( make_nvp("shape", make_carray_view(shape.data(), shape.size())) );
	else
		ar( make_nvp("shape", shape) );
	// size
	ar( make_size_tag(t.size()) );
	// because of strides we can only save content element by element
	for(Eigen::Index i = 0; i < t.size(); ++i)
		ar(t.coeff(i));
}

template <class Archive, class Derived>
auto load(Archive& ar, Eigen::MapBase<Derived, Eigen::AccessorLevels::WriteAccessors>& t) -> void {
	// read shape
	std::array<Eigen::Index, 2> shape;
	if constexpr(cereal::traits::is_text_archive<Archive>::value)
		ar( make_nvp("shape", make_carray_view(shape.data(), shape.size())) );
	else
		ar( make_nvp("shape", shape) );
	// read size
	Eigen::Index sz;
	ar( make_size_tag(sz) );
	if(t.size() < sz) throw blue_sky::error("Size of Eigen::Map target is less than required");
	// read data
	for(Eigen::Index i = 0; i < sz; ++i)
		ar(t.coeff(i));
}

} // eof cereal

