/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../fwd.h"
#include "../common.h"
#include "arrbase.h"
#include <iterator>

namespace blue_sky {

/*-----------------------------------------------------------------
 * Implement array interface around container stored in shared_ptr
 * Thus, array data is shared
 *----------------------------------------------------------------*/
template< class T, class array_t >
class BS_API bs_arrbase_shared_impl : public bs_arrbase< T > {
public:
	// traits for bs_array
	using arrbase = bs_arrbase< T >;
	// here container means std::shared_ptr< container > to deal with bs_array
	using container = std::shared_ptr< array_t >;
	using bs_array_base = bs_arrbase_shared_impl< T, array_t >;

	using typename arrbase::sp_arrbase;
	using sp_array_shared = std::shared_ptr< bs_array_base >;

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

	// default, copy and move ctors
	bs_arrbase_shared_impl() = default;
	bs_arrbase_shared_impl(const bs_arrbase_shared_impl&) = default;
	bs_arrbase_shared_impl(bs_arrbase_shared_impl&&) = default;

	// special ctor if container is passed
	bs_arrbase_shared_impl(const container& c) {
		init_inplace(c);
	}

	// perfect forwarding ctor forwards all args to container
	// specifically treats first argument to allow vonstructors above
	template<
		typename First, typename... Args,
		typename = std::enable_if_t<
			!std::is_base_of< bs_arrbase_shared_impl, std::decay_t< First > >::value &&
			!std::is_base_of< container, std::decay_t< First > >::value
		>
	>
	bs_arrbase_shared_impl(First&& first, Args&&... args)
		: shared_data_(std::make_shared< array_t >(
			std::forward< First >(first), std::forward< Args >(args)...
		))
	{}

	// perfect forwarding init
	template< typename... Args >
	void init(Args&&... args) {
		shared_data_ = std::make_shared< array_t >(std::forward< Args >(args)...);
	}

	// perfect forwarding create
	template< typename... Args >
	static sp_array_shared create(Args&&... args) {
		return std::make_shared< bs_arrbase_shared_impl >(std::forward< Args >(args)...);
	}

	// can be called in any time to switch container
	void init_inplace(const container& c) {
		if(shared_data_.get() != c.get())
			shared_data_ = c;
	}

	// implement bs_arrbase interface
	size_type size() const {
		if(shared_data_)
			return shared_data_->size();
		return 0;
	}

	void resize(size_type n) {
		if(shared_data_ && size() != n)
			shared_data_->resize(n);
	}

	void resize(size_type n, value_type init) {
		if(shared_data_ && size() != n)
			shared_data_->resize(n, init);
	}

	pointer data() {
		if(shared_data_)
			return shared_data_->data();
		return nullptr;
	}

	const_pointer data() const {
		if(shared_data_)
			return shared_data_->data();
		return nullptr;
	}

	// explicitly make array copy
	sp_arrbase clone() const {
		return create(*this);
	}

	// reference to rhs array
	// disable -- compiler-generated operator= is fine
	//bs_arrbase_shared_impl& operator=(const bs_arrbase_shared_impl& rhs) {
	//	shared_data_ = rhs.shared_data_;
	//	return *this;
	//}

	void swap(bs_arrbase_shared_impl& rhs) {
		std::swap(shared_data_, rhs.shared_data_);
	}

	friend bool operator==(const bs_arrbase_shared_impl& lhs, const bs_arrbase_shared_impl& rhs) {
		return lhs.data() == rhs.data();
	}

	friend bool operator<(const bs_arrbase_shared_impl& lhs, const bs_arrbase_shared_impl& rhs) {
		return lhs.data() < rhs.data();
	}

	// access to container
	container get_container() const {
		return shared_data_;
	}

protected:
	// if shared array owns buffer here's real buffer handler
	// otherwise NULL
	container shared_data_;
};

// bs_array_shared uses std::vector as underlying container
template< class T > using bs_array_shared = bs_arrbase_shared_impl< T, std::vector< T > >;

}   // eof blue_sky

