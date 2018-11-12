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

namespace cereal {

/*-----------------------------------------------------------------------------
 *  `Array` or `Matrix`
 *-----------------------------------------------------------------------------*/
template <typename Archive, typename Derived>
auto save(Archive& ar, Eigen::PlainObjectBase<Derived> const& t) -> void {
	using ArrT = Eigen::PlainObjectBase<Derived>;
	if(ArrT::RowsAtCompileTime == Eigen::Dynamic) ar(make_nvp("rows", t.rows()));
	if(ArrT::ColsAtCompileTime == Eigen::Dynamic) ar(make_nvp("cols", t.cols()));
	ar( make_nvp("data", make_carray_view(t.data(), t.size())) );
}

template <typename Archive, typename Derived>
auto load(Archive& ar, Eigen::PlainObjectBase<Derived>& t) -> void {
	using ArrT = Eigen::PlainObjectBase<Derived>;
	Eigen::Index rows = ArrT::RowsAtCompileTime, cols = ArrT::ColsAtCompileTime;
	if(rows == Eigen::Dynamic) ar(make_nvp("rows", rows));
	if(cols == Eigen::Dynamic) ar(make_nvp("cols", cols));
	t.resize(rows, cols);
	ar( make_nvp("data", make_carray_view(t.data(), t.size())) );
}

/*-----------------------------------------------------------------------------
 *  `Map` or `Block`
 *-----------------------------------------------------------------------------*/
template <typename Archive, typename Derived, Eigen::AccessorLevels A>
auto save(Archive& ar, Eigen::MapBase<Derived, A> const& t) -> void {
	// size
	ar( make_size_tag(t.size()) );
	// because of strides we can only save content element by element
	for(Eigen::Index i = 0; i < t.size(); ++i)
		ar(t.coeff(i));
}

template <class Archive, class Derived>
auto load(Archive& ar, Eigen::MapBase<Derived, Eigen::AccessorLevels::WriteAccessors>& t) -> void {
	// read size
	Eigen::Index sz;
	ar( make_size_tag(sz) );
	if(sz < t.size()) throw blue_sky::error("Size of target Eigen::Map is less that required");
	// read data
	for(Eigen::Index i = 0; i < sz; ++i)
		ar(t.coeff(i));
}

} // eof cereal

