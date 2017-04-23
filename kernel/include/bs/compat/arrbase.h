/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../common.h"
#include <iterator>

namespace blue_sky {

/*-----------------------------------------------------------------------------
 *  bs_arrbase -- base class of BlueSky arrays
 *-----------------------------------------------------------------------------*/

/// @brief Base class of all BlueSky arrays with Python numpy support
/// Expected that underlying container is std::vector-like, i.e.
/// all elements are stored in continuos memory segment
template< class T >
class bs_arrbase {
public:
	typedef std::shared_ptr< bs_arrbase > sp_arrbase;

	typedef T           value_type;
	typedef std::size_t size_type;
	typedef size_type   key_type;

	typedef T*                                      pointer;
	typedef T&                                      reference;
	typedef T const*                                const_pointer;
	typedef T const&                                const_reference;
	typedef pointer                                 iterator;
	typedef const_pointer                           const_iterator;
	typedef std::reverse_iterator< iterator >       reverse_iterator;
	typedef std::reverse_iterator< const_iterator > const_reverse_iterator;

	/// @brief Obtain array size
	///
	/// @return number of elements contained in array
	virtual size_type size() const = 0;
	
	/// @brief implement resizing of array
	///
	/// @param new_size -- new size of array
	virtual void resize(size_type new_size) = 0;


	/// @brief resize array with new initialization of new elements
	///
	/// @param new_size -- new size of array
	/// @param init value of ew elements
	virtual void resize(size_type new_size, value_type init) = 0;

	/// @brief gain access to data buffer
	///
	/// @return pointer to raw data
	virtual pointer data() = 0;

	virtual const_pointer data() const {
		return const_cast< bs_arrbase* >(this)->data();
	}

	/// @brief truely copy array data
	///
	/// @return smart_ptr to array with copied data
	virtual sp_arrbase clone() const = 0;

	/// @brief check if array is empty
	///
	/// @return true if array contains no elements, false otherwise
	virtual bool empty() const {
		return (size() == 0);
	}

	/// @brief Subscripting operator
	/// @param key item key
	/// 
	/// @return modifiable reference to element
	virtual reference operator[](const key_type& key) {
		return *(data() + key);
	}

	/// @brief Subscripting operator
	/// @param key item key
	/// 
	/// @return modifiable reference to element
	virtual const_reference operator[](const key_type& key) const {
		return *(data() + key);
	}

	/// @brief Items access function (r/w) - syntax sugar for accessing via pointer
	///
	/// @param key item key
	///
	/// @return modifiable reference to element
	reference ss(const key_type& key) {
		return operator[](key);
	}

	/// @brief Items access function (r) - syntax sugar for accessing via pointer
	///
	/// @param key item key
	///
	/// @return const reference to element
	const_reference ss(const key_type& key) const {
		return operator[](key);
	}

	virtual void clear() {
		resize(0);
	}

	virtual iterator begin() {
		return data();
	}
	virtual const_iterator begin() const {
		return data();
	}

	virtual iterator end() {
		return data() + size();
	}

	virtual const_iterator end() const {
		return data() + size();
	}

	virtual reverse_iterator rbegin() {
		return reverse_iterator(end());
	}
	virtual const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}

	virtual reverse_iterator rend() {
		return reverse_iterator(begin());
	}
	virtual const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}

	virtual void assign(const value_type& v) {
		std::fill(begin(), end(), v);
	}

	/// @brief iterator version of assign - alias for std::copy
	/// no bound-checking performed, make sure range(start, finish) <= dest array size
	/// @tparam input_iterator
	/// @param start begin of source range
	/// @param finish end of source range
	template< class input_iterator >
	void assign(const input_iterator start, const input_iterator finish) {
		std::copy(start, finish, this->begin());
	}

	// get first & last elements
	reference back() {
		return *(end() - 1);
	}
	const_reference back() const {
		return *(end() - 1);
	}

	reference front() {
		return *begin();
	}
	const_reference front() const {
		return *begin();
	}

	/// @brief empty virtual destructor
	virtual ~bs_arrbase() {};
};

/*-----------------------------------------------------------------
 * bs_arrbase_impl = bs_arrbase + custom container (minimum overloads)
 *----------------------------------------------------------------*/
template< class T, class array_t >
class bs_arrbase_impl : public array_t, public bs_arrbase< T > {
public:
    // traits for bs_array
	using container = array_t;
	using arrbase = bs_arrbase< T >;
	using bs_array_base = bs_arrbase_impl< T, array_t >;
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
	using arrbase::rbegin;
	using arrbase::rend;
	using arrbase::clear;
	using arrbase::back;
	using arrbase::front;

	// arrbase doesn't need to be constructed, so make perfect forwarding ctor to array_t
	template < typename... Args >
	bs_arrbase_impl(Args&&... args) : array_t(std::forward< Args >(args)...) {}

	size_type size() const {
		return static_cast< size_type >(container::size());
	}

	void resize(size_type new_size) {
		container::resize(new_size);
	}

	void resize(size_type new_size, value_type init) {
		container::resize(new_size, init);
	}

	pointer data() {
		if(this->size())
			return &this->operator[](0);
		else return nullptr;
	}

	void clear() {
		container::clear();
	}

	bool empty() const {
		return container::empty();
	}

	sp_arrbase clone() const {
		return std::make_shared< bs_arrbase_impl >(*this);
	}

	reference operator[](const key_type& key) {
		return container::operator[](key);
	}

	const_reference operator[](const key_type& key) const {
		return container::operator[](key);
	}

	void swap(bs_arrbase_impl& rhs) {
		container::swap(rhs);
	}
};

}	// namespace blue-sky

