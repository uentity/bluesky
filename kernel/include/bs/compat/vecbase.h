/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Vector-like BlueSky array declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "arrbase.h"

namespace blue_sky {

/*-----------------------------------------------------------------------------
 *  bs_vecbase -- base class of BlueSky vectors
 *-----------------------------------------------------------------------------*/
// difference from arrbase is insert and erase operations
template< class T >
class bs_vecbase {
public:
	using arrbase    = bs_arrbase< T >;
	using key_type   = typename arrbase::key_type;
	using value_type = typename arrbase::value_type;
	using size_type  = typename arrbase::size_type;
	using iterator   = typename arrbase::iterator;

	virtual bool insert(const key_type& key, const value_type& value) = 0;
	virtual bool insert(const value_type& value) = 0;
	virtual iterator insert(iterator pos, const value_type& v) = 0;
	virtual void insert(iterator pos, size_type n, const value_type& v) = 0;

	virtual void erase(const key_type& key) = 0;
	virtual iterator erase(iterator pos) = 0;
	virtual iterator erase(iterator start, iterator finish) = 0;

	virtual void clear() = 0;
	virtual void reserve(size_type sz) = 0;

	virtual void push_back(const value_type&) = 0;
	virtual void pop_back() = 0;

	/// @brief empty destructor
	virtual ~bs_vecbase() {};
};

/*-----------------------------------------------------------------
 * bs_vecbase_impl implements bs_vecbase and bs_arrbase iface using custom vector-like container
 *----------------------------------------------------------------*/
template< class T, class vector_t >
class bs_vecbase_impl : public bs_arrbase_impl< T, vector_t >, public bs_vecbase< T > {
public:
	using base_t        = bs_arrbase_impl< T, vector_t >;
    // traits for bs_array
	using container     = vector_t;
	using arrbase       = typename base_t::arrbase;
	using bs_array_base = bs_vecbase_impl< T, vector_t >;
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

	using container::empty;

	// vecbase doesn't need to be constructed, so make perfect forwarding ctor to bs_array_impl
	template < typename... Args, typename = meta::enable_pf_ctor_to<bs_vecbase_impl, vector_t, Args...> >
	bs_vecbase_impl(Args&&... args) : base_t(std::forward< Args >(args)...) {}

	bool insert(const key_type& key, const value_type& value) {
		if(key > this->size()) return false;
		container::insert(container::begin() + key, value);
		return true;
	}

	bool insert(const value_type& value) {
		container::push_back(value);
		return true;
	}

	iterator insert(iterator pos, const value_type& v) {
		return iter2ptr(container::insert(ptr2iter(pos), v));
	}

	void insert(iterator pos, size_type n, const value_type& v) {
		container::insert(ptr2iter(pos), n, v);
	}

	template< class inp_iterator>
	void insert(iterator pos, inp_iterator from, inp_iterator to) {
		container::insert(ptr2iter(pos), from, to);
	}

	void erase(const key_type& key)	{
		container::erase(container::begin() + key);
	}

	iterator erase(iterator pos) {
		return iter2ptr(container::erase(ptr2iter(pos)));
	}

	iterator erase(iterator start, iterator finish) {
		return iter2ptr(container::erase(ptr2iter(start), ptr2iter(finish)));
	}

	void clear() {
		container::clear();
	}

	virtual void push_back(const value_type& v) {
		container::push_back(v);
	}

	virtual void pop_back() {
		container::pop_back();
	}

	void reserve(size_type sz) {
		container::reserve(sz);
	}

	sp_arrbase clone() const {
		return std::make_shared< bs_vecbase_impl >(*this);
	}

	void swap(bs_vecbase_impl& rhs) {
		base_t::swap(rhs);
	}

private:
	inline typename container::iterator ptr2iter(const iterator& pos) {
		return container::begin() + (pos - this->begin());
	}
	inline typename container::const_iterator ptr2iter(const const_iterator& pos) const {
		return container::begin() + (pos - this->begin());
	}

	inline iterator iter2ptr(const typename container::iterator& pos) {
		return this->begin() + (pos - container::begin());
	}
	inline const_iterator iter2ptr(const typename container::const_iterator& pos) const {
		return this->begin() + (pos - container::begin());
	}
};

} 	// eof blue_sky

