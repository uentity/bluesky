// This file is part of BlueSky
//
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#ifndef _BS_ARRBASE_H_
#define _BS_ARRBASE_H_

#include "bs_common.h"
#include <iterator>

namespace blue_sky {

/*-----------------------------------------------------------------------------
 *  bs_arrbase -- base class of BlueSky arrays
 *-----------------------------------------------------------------------------*/

/// @brief Base class of all BlueSky arrays with Python numpy support
/// Expected that underlying container is std::vector-like, i.e.
/// all elements are stored in continuos memory segment
template< class T >
class BS_API bs_arrbase : virtual public bs_refcounter {
public:
	typedef bs_arrbase< T > this_t;
	typedef smart_ptr < this_t, true > sp_arrbase;

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

	/// @brief gain access to data buffer
	///
	/// @return pointer to raw data
	virtual pointer data() = 0;

	virtual const_pointer data() const {
		return const_cast< this_t* >(this)->data();
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

	/// @brief assign from array of castable type
	///
	/// @tparam R type of rhs array
	/// @param rhs source of assignment
	template< class R >
	void assign(const bs_arrbase< R >& rhs) {
		size_type n = rhs.size();
		this->resize(n);
		if((void*)this->begin() != (void*)rhs.begin())
			std::copy(rhs.begin(), rhs.begin() + std::min(n, this->size()), this->begin());
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
class BS_API bs_arrbase_impl : public bs_arrbase< T >, public array_t {
public:
	typedef array_t container;
	typedef bs_arrbase< T > arrbase;
	typedef typename arrbase::sp_arrbase sp_arrbase;
	typedef bs_arrbase_impl< T, array_t > bs_array_base;

	// inherited from bs_arrbase class
	typedef typename arrbase::value_type value_type;
	typedef typename arrbase::key_type   key_type;
	typedef typename arrbase::size_type  size_type;

	typedef typename arrbase::pointer                pointer;
	typedef typename arrbase::reference              reference;
	typedef typename arrbase::const_pointer          const_pointer;
	typedef typename arrbase::const_reference        const_reference;
	typedef typename arrbase::iterator               iterator;
	typedef typename arrbase::const_iterator         const_iterator;
	typedef typename arrbase::reverse_iterator       reverse_iterator;
	typedef typename arrbase::const_reverse_iterator const_reverse_iterator;

	using arrbase::begin;
	using arrbase::end;
	using arrbase::rbegin;
	using arrbase::rend;
	using arrbase::assign;
	using arrbase::clear;
	using arrbase::back;
	using arrbase::front;

	// default ctor
	bs_arrbase_impl() {}

	// ctor for constructing from container copy
	bs_arrbase_impl(const container& c) : array_t(c) {}

	// given size & fill value
	bs_arrbase_impl(size_type sz, const value_type& v = value_type()) : array_t(sz, v) {}

	size_type size() const {
		return static_cast< size_type >(container::size());
	}

	void resize(size_type new_size) {
		container::resize(new_size);
	}

	pointer data() {
		if(this->size())
			return &this->operator[](0);
		else return 0;
	}

	sp_arrbase clone() const {
		sp_arrbase res = new bs_arrbase_impl(this->size());
		std::copy(begin(), end(), res->begin());
		return res;
	}

	reference operator[](const key_type& key) {
		return container::operator[](key);
	}

	const_reference operator[](const key_type& key) const {
		return container::operator[](key);
	}

	void dispose() const {
		delete this;
	}

	void swap(bs_arrbase_impl& rhs) {
		container::swap(rhs);
	}
};

}	// namespace blue-sky
#endif	// file guard

