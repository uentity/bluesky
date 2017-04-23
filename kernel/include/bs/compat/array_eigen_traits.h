/// @file
/// @author uentity
/// @date 02.11.2016
/// @brief Traits for BS array based on Eigen matrix backend
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "arrbase.h"
#include <algorithm>
#include <Eigen/Dense>

namespace blue_sky {

// represents row
template< class T >
class eigen_traits : public Eigen::Array< T, 1, Eigen::Dynamic >, public bs_arrbase< T > {
public:
	// traits for bs_array
	using container = Eigen::Array< T, 1, Eigen::Dynamic >;
	using arrbase = bs_arrbase< T >;
	using bs_array_base = eigen_traits;
	using typename arrbase::sp_arrbase;

	// inherited from bs_arrbase class
	using typename arrbase::value_type;
	using typename arrbase::key_type;
	using typename arrbase::size_type;

	using typename arrbase::pointer;
	using typename arrbase::reference;
	using typename arrbase::const_pointer;
	using typename arrbase::const_reference;
	using typename arrbase::iterator;
	using typename arrbase::const_iterator;
	using typename arrbase::reverse_iterator;
	using typename arrbase::const_reverse_iterator;

	using arrbase::begin;
	using arrbase::end;

	// arrbase doesn't need to be constructed, so make perfect forwarding ctor to array_t
	//template <
	//	typename Arg1, typename... Args,
	//	typename = std::enable_if_t<
	//		!std::is_same< std::decay_t<Arg1>, const_iterator >::value
	//	>
	//>
	//eigen_traits(Arg1&& arg1, Args&&... args)
	//	: container(std::forward<Arg1>(arg1), std::forward< Args >(args)...)
	//{}

	using container::container;

	eigen_traits(size_type sz, value_type init) : container(sz) {
		(container&)(*this) = init;
		//std::fill(begin(), end(), init);
	}

	// constructor from pair of iterators
	eigen_traits(const const_iterator& from, const const_iterator& to)
		: container(std::max< decltype(to - from) >(to - from, 0))
	{
		if(to > from)
			std::copy(from, to, begin());
	}

	eigen_traits() = default;

	size_type size() const {
		return static_cast< size_type >(container::size());
		//return static_cast< size_type >(container::cols() * container::rows());
	}

	void resize(size_type new_size) {
		container::resize(new_size);
	}

	void resize(size_type new_size, value_type init) {
		auto old_sz = size();
		container::conservativeResize(new_size);
		if(new_size > old_sz)
			std::fill(begin() + old_sz, begin() + new_size, init);
	}

	pointer data() {
		return container::data();
	}

	//void clear() {
	//	container::clear();
	//}

	//bool empty() const {
	//	return size() == 0;
	//}

	sp_arrbase clone() const {
		return std::make_shared< eigen_traits >(*this);
	}

	reference operator[](const key_type& key) {
		return container::operator()(key);
	}

	const_reference operator[](const key_type& key) const {
		return container::operator()(key);
	}

	void swap(eigen_traits& rhs) {
		container::swap(rhs);
	}
};

} /* namespace blue_sky */

