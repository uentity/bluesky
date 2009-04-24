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
#include "bs_object_base.h"
#include "bs_kernel.h"
#include <iterator>

namespace blue_sky {

/*-----------------------------------------------------------------------------
 *  bs_arrbase class - base class of BlueSky arrays
 *-----------------------------------------------------------------------------*/
template< class T >
class bs_arrbase : public objbase {
public:
	typedef T value_type;
	typedef ulong size_type;
	typedef size_type key_type;

	typedef value_type& reference;
	typedef const value_type& const_reference;

	// abstract class that declares typeless random access iterator iface
	template< class val_t >
	class iterator_backend {
	public:
		typedef iterator_backend< val_t > this_t;
		typedef st_smart_ptr< this_t > this_ptr;

		typedef std::iterator< std::random_access_iterator_tag, val_t > iter_traits;
		typedef typename iter_traits::value_type value_type;
		typedef typename iter_traits::pointer pointer;
		typedef typename iter_traits::reference reference;
		typedef typename iter_traits::difference_type diff_t;

		// used to initialize backend
		//virtual void assign(const this_t&) = 0;
		virtual this_ptr clone() = 0;
		// dtor
		virtual ~iterator_backend() {}

		// public iterator interface starts here
		virtual reference get_ref() const = 0;
		virtual reference item(diff_t ind) const = 0;

		virtual void operator++() = 0;
		virtual void operator--() = 0;

		virtual void operator+=(diff_t n) = 0;
		virtual void operator-=(diff_t n) = 0;

		// get distance between iterators
		virtual diff_t distance(const this_t&) const = 0;

		// comparison
		virtual bool operator==(const this_t&) const = 0;
		virtual bool operator<(const this_t&) const = 0;
		virtual bool operator>(const this_t&) const = 0;
	};

	// non-virtual class converting backend to stl random iterator interface
	template< class val_t >
	class iterator_t : public std::iterator< std::random_access_iterator_tag, val_t > {
	public:
		// nessesary typedefs
		typedef iterator_t< val_t > this_t;
		typedef iterator_backend< val_t > backend_t;
		typedef typename backend_t::this_ptr backend_ptr;

		typedef std::iterator< std::random_access_iterator_tag, val_t > iter_traits;
		typedef typename iter_traits::value_type value_type;
		typedef typename iter_traits::pointer pointer;
		typedef typename iter_traits::reference reference;
		typedef typename iter_traits::difference_type diff_t;

		// default ctor -- needed for uninitialized declarations
		iterator_t()
			: ibe_(NULL)
		{}

		// standard ctor
		iterator_t(const backend_ptr& ibe)
			: ibe_(ibe)
		{
			assert(ibe_ == NULL);
		}

		// copy ctor
		iterator_t(const iterator_t& i)
			: ibe_(i.ibe_->clone())
		{
			assert(ibe_ == NULL);
		}

		// assignemnt operator
		this_t& operator=(const this_t& i) {
			// through swap
			this_t(i).swap(*this);
			return *this;
		}

		// public iterator interface starts here
		reference operator*() const {
			return ibe_->get_ref();
		}
		pointer operator->() const {
			return &ibe_->get_ref();
		}
		reference operator[](diff_t ind) const {
			return ibe_->item(ind);
		}

		this_t& operator++() {
			++(*ibe_);
			return *this;
		}
		this_t operator++(int) {
			this_t tmp(*this);
			this->operator++();
			return tmp;
		}

		this_t& operator--()  {
			--(*ibe_);
			return *this;
		}
		this_t operator--(int) {
			this_t tmp(*this);
			this->operator--();
			return tmp;
		}

		this_t& operator+=(diff_t n) {
			*ibe_ += n;
		}
		this_t operator+(diff_t n) const {
			this_t tmp(*this);
			tmp += n;
			return tmp;
		}

		this_t& operator-=(diff_t n) {
			*ibe_ -= n;
		}
		this_t operator-(diff_t n) const {
			this_t tmp(*this);
			tmp -= n;
			return tmp;
		}

		// return distance between iterators
		diff_t operator-(const this_t& i) const {
			return ibe_->distance(i.ibe_);
		}

		// comparison operators
		bool operator==(const this_t& i) const {
			return *ibe_ == *i.ibe_;
		}
		bool operator!=(const this_t& i) const {
			return !(*this == i);
		}

		bool operator<(const this_t& i) const {
			return *ibe_ < *i.ibe_;
		}
		bool operator>(const this_t& i) const {
			return *ibe_ > *i.ibe_;
		}
		bool operator<=(const this_t& i) const {
			return !(*this > i);
		}
		bool operator>=(const this_t& i) const {
			return !(*this < i);
		}

	private:
		backend_ptr ibe_;

		// swaps 2 iterator_t
		void swap(this_t& i) {
			std::swap(ibe_, i.ibe_);
		}
	};

	// CRTP wrapper for converting any particular iterator into iterator_t
	// just forwards calls to base ra_iterator class
	template< class val_t, class ra_iterator >
	class iterator_wrapper : public iterator_backend< val_t >, public ra_iterator {
	public:
		
		// nessesary typedefs
		typedef iterator_wrapper< val_t, ra_iterator > this_t;
		typedef iterator_backend< val_t > base_t;
		typedef typename base_t::this_ptr base_ptr;

		typedef typename base_t::value_type value_type;
		typedef typename base_t::pointer pointer;
		typedef typename base_t::reference reference;
		typedef typename base_t::diff_t diff_t;

		// default ctor
		iterator_wrapper() {}
		// construct from ra_iterator
		iterator_wrapper(const ra_iterator& i)
			: ra_iterator(i)
		{}

		// copy ctors
		iterator_wrapper(const this_t& i)
			: ra_iterator(i)
		{}

		base_ptr clone() {
			return new this_t(*this);
		}

		// public iterator interface starts here
		reference get_ref() const {
			return ra_iterator::operator*();
		}
		reference item(diff_t ind) const {
			return ra_iterator::operator[](ind);
		}

		void operator++() {
			ra_iterator::operator++();
		}

		void operator--() {
			ra_iterator::operator--();
		}

		void operator+=(diff_t n) {
			ra_iterator::operator+=(n);
		}
		void operator-=(diff_t n) {
			ra_iterator::operator-=(n);
		}

		// return distance between iterators
		diff_t distance(const base_t& i) const {
			return *(ra_iterator*)this - cast_base(i);
		}

		// comparison operators
		// NOTE that comparison will only work if right-hand operand is of type ra_iterator or this_t
		bool operator==(const base_t& i) const {
			return (*this == cast_base(i));
		}
		bool operator<(const base_t& i) const {
			return (*this < cast_base(i));
		}
		bool operator>(const base_t& i) const {
			return (*this > cast_base(i));
		}

	private:
		inline const ra_iterator& cast_base(const base_t& i) const {
			if(BS_GET_TI(*this) == BS_GET_TI(i)) {
				const ra_iterator* tmp = reinterpret_cast< const ra_iterator* >(&i);
				return *tmp;
				//return static_cast< const ra_iterator& >(i);
			}
			else
				throw bs_exception("bs_arrbase::iterator_wrapper", "Iterators types mismatch during comparison");
		}
	};

	// construct iterator & const_iterator
	typedef iterator_t< value_type > iterator;
	typedef iterator_t< const value_type > const_iterator;

	/// @brief Obtain array size
	/// 
	/// @return number of elements contained in array
	virtual size_type size() const = 0;

	/// @brief Items access operator (r/w)
	/// 
	/// @param key item key
	/// 
	/// @return modifyable refernce to element
	virtual reference operator[](const key_type& key) = 0;

	/// @brief Items access operator (r)
	/// 
	/// @param key item key
	/// 
	/// @return const refernce to element
	virtual const_reference operator[](const key_type& key) const = 0;
	
	// begin-end support for array iteration
	virtual iterator begin() = 0;
	virtual const_iterator begin() const = 0;
	virtual iterator end() = 0;
	virtual const_iterator end() const = 0;

	//! empty destructor
	virtual ~bs_arrbase() {};

	BS_COMMON_DECL_T_MEM(bs_arrbase);
	BS_LOCK_THIS_DECL(bs_arrbase);
	
	//BLUE_SKY_TYPE_DECL_T_MEM(bs_arrbase, objbase, "bs_arrbase",
	//	"Base class for BlueSky continuous array of values of the same type indexed by integral type", "");
};

template< class T >
bs_arrbase< T >::bs_arrbase(bs_type_ctor_param param)
	: bs_refcounter(), objbase(param)
{}

// copy ctor
template< class T >
bs_arrbase< T >::bs_arrbase(const bs_arrbase& a)
	: bs_refcounter(), objbase(a)
{}

}	// namespace blue-sky
#endif	// file guard

