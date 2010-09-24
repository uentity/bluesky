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

	typedef T value_type;
	typedef std::size_t size_type;
	typedef size_type key_type;

	typedef T* pointer;
	typedef T& reference;
	typedef T const* const_pointer;
	typedef T const& const_reference;
	typedef pointer        iterator;
	typedef const_pointer  const_iterator;
	typedef std::reverse_iterator< iterator >             reverse_iterator;
	typedef std::reverse_iterator< const_iterator > const_reverse_iterator;

	/// @brief Obtain array size
	///
	/// @return number of elements contained in array
	virtual size_type size() const = 0;

	virtual bool empty() const {
		return (size() == 0);
	}

	/// @brief Subscripting operator
	/// Forward call to ss(key)
	/// @param key item key
	/// 
	/// @return modifiable reference to element
	virtual reference operator[](const key_type& key) = 0;

	/// @brief Items access function (r/w) - syntax sugar for accessing via pointer
	///
	/// @param key item key
	///
	/// @return modifiable reference to element
	reference ss(const key_type& key) {
		return operator[](key);
	}

	/// @brief Subscripting operator
	/// Forward call to ss(key)
	/// @param key item key
	/// 
	/// @return modifiable reference to element
	virtual const_reference operator[](const key_type& key) const = 0;

	/// @brief Items access function (r) - syntax sugar for accessing via pointer
	///
	/// @param key item key
	///
	/// @return const reference to element
	const_reference ss(const key_type& key) const {
		return operator[](key);
	}

	virtual void resize(size_type new_size) = 0;

	virtual void clear() {
		resize(0);
	}

	virtual iterator begin() {
		return &ss(0);
	}
	virtual const_iterator begin() const {
		return &ss(0);
	}

	virtual iterator end() {
		return &ss(0) + size();
	}
	virtual const_iterator end() const {
		return &ss(0) + size();
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

	// make array with copied data
	virtual sp_arrbase clone() const = 0;

	virtual void assign(const value_type& v) {
		std::fill(begin(), end(), v);
	}

	// assign for different type array
	template< class R >
	void assign(const bs_arrbase< R >& rhs) {
		size_type n = rhs.size();
		if(this->size() != n) this->resize(n);
		if(this->size() && this->begin() != rhs.begin())
			std::copy(rhs.begin(), rhs.begin() + std::min(n, this->size()), this->begin());
	}

	/// @brief empty destructor
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
	typedef typename arrbase::key_type key_type;
	typedef typename arrbase::size_type size_type;

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

	// default ctor
	bs_arrbase_impl() {}

	// ctor for constructing from container copy
	bs_arrbase_impl(const container& c) : array_t(c) {}

	size_type size() const {
		return static_cast< size_type >(container::size());
	}

	reference operator[](const key_type& key) {
		return container::operator[](key);
	}

	const_reference operator[](const key_type& key) const {
		return container::operator[](key);
	}

	void resize(size_type new_size) {
		container::resize(new_size);
	}

	sp_arrbase clone() const {
		return new bs_arrbase_impl(*this);
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

