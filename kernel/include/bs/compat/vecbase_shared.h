/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Shared vector interface implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "arrbase_shared.h"
#include "vecbase.h"

namespace blue_sky {

/*-----------------------------------------------------------------
 * Make array around plain buffer optionally held by bs_arrbase-compatible container 
 *----------------------------------------------------------------*/
template< class T, class vector_t >
class BS_API bs_vecbase_shared_impl :
	// bind bs_vecbase< T > interface to vector_t container
	// this allows to utilise vecbase implementation directly in shared_data_
	public bs_arrbase_shared_impl< T, bs_vecbase_impl< T, vector_t > >,
	public bs_vecbase< T >
{
public:
	using base_t = bs_arrbase_shared_impl< T, bs_vecbase_impl< T, vector_t > >;
	// traits for bs_array
	using arrbase = typename base_t::arrbase;
	using typename base_t::container;
	using bs_array_base = bs_vecbase_shared_impl< T, vector_t >;

	typedef std::shared_ptr< bs_array_base > sp_vector_shared;
	using typename arrbase::sp_arrbase;

	//typedef bs_vecbase< T > vecbase_t;

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

	// inherit all constructors from base class
	using base_t::base_t;

	// perfect forwarding create
	template< typename... Args >
	static sp_vector_shared create(Args&&... args) {
		return std::make_shared< bs_vecbase_shared_impl >(std::forward< Args >(args)...);
	}

	void push_back(const value_type& v) {
		shared_data_->push_back(v);
	}

	void pop_back() {
		shared_data_->pop_back();
	}

	iterator insert(iterator pos, const value_type& v) {
		return shared_data_->insert(pos, v);
	}
	void insert(iterator pos, size_type n, const value_type& v) {
		shared_data_->insert(pos, n, v);
	}

	template< class input_iterator >
	void insert(iterator pos, input_iterator start, input_iterator finish) {
		shared_data_->insert(pos, start, finish);
	}

	iterator erase(iterator pos) {
		return shared_data_->erase(pos);
	}
	iterator erase(iterator start, iterator finish) {
		return shared_data_->erase(start, finish);
	}

	// overloads from bs_vecbase
	bool insert(const key_type& key, const value_type& value) {
		return shared_data_->insert(key, value);
	}

	bool insert(const value_type& value) {
		return shared_data_->insert(value);
	}

	void erase(const key_type& key)	{
		shared_data_->erase(key);
	}

	void clear() {
		shared_data_->clear();
	}

	void reserve(size_type sz) {
		shared_data_->reserve(sz);
	}

	// explicitly make array copy
	sp_arrbase clone() const {
		return create(*this);
	}

private:
	using base_t::shared_data_;
};

// bs_vector_shared uses std::vector as underlying container
template< class T > using bs_vector_shared = bs_vecbase_shared_impl< T, std::vector< T > >;

} 	// eof blue_sky

