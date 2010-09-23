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

#ifndef BS_ARRAY_SHARED_529LXCQA
#define BS_ARRAY_SHARED_529LXCQA

#include "bs_fwd.h"
#include "bs_common.h"
#include "bs_arrbase.h"
#include "bs_kernel.h"
#include <iterator>

namespace blue_sky {

/*-----------------------------------------------------------------
 * Make array around plain buffer optionally held by bs_arrbase-compatible container
 *----------------------------------------------------------------*/
template< class T >
class BS_API bs_array_shared : public bs_arrbase< T > {
public:
	// traits for bs_array
	typedef bs_arrbase< T > arrbase;
	//typedef container_t< T > container;
	// here container means smart_ptr< container > to deal with bs_array
	typedef smart_ptr< bs_arrbase< T > > container;
	typedef bs_array_shared< T > bs_array_base;

	typedef smart_ptr< bs_array_base, true > sp_array_shared;
	typedef typename arrbase::sp_arrbase sp_arrbase;
	typedef bs_array< T, vector_traits > def_cont_impl;
	typedef Loki::Type2Type< def_cont_impl > def_cont_tag;

	typedef T              value_type;
	typedef std::size_t    size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T*             pointer;
	typedef T const*       const_pointer;
	typedef T&             reference;
	typedef T const&       const_reference;
	typedef pointer        iterator;
	typedef const_pointer  const_iterator;

	typedef std::reverse_iterator< iterator >       reverse_iterator;
	typedef std::reverse_iterator< const_iterator > const_reverse_iterator;

	template< class container >
	void init(Loki::Type2Type< container >, size_type n = 0, const value_type& v = value_type()) {
		smart_ptr< container > c = BS_KERNEL.create_object(container::bs_type());
		data_ = NULL; size_ = 0;
		if(!c) return;

		if(n)
			c->init(n, v);
		if((size_ = c->size()))
			data_ = &c->ss(0);
		buf_holder_ = c;
	}

	template< class input_iterator, class container >
	void init(Loki::Type2Type< container >, input_iterator start, input_iterator finish) {
		smart_ptr< container > c = BS_KERNEL.create_object(container::bs_type());
		size_ = 0; data_ = NULL;
		if(!c) return;

		c->init(start, finish);
		if((size_ = c->size()))
			data_ = &c->ss(0);
		buf_holder_ = c;
	}

	// ctors, array ownes data
	bs_array_shared(size_type n = 0, const value_type& v = value_type()) {
		init(def_cont_tag(), n, v);
	}

	template< class container >
	bs_array_shared(
			size_type n = 0,
			const value_type& v = value_type(),
			const Loki::Type2Type< container >& t = def_cont_tag()
			)
	{
		init(t, n, v);
	}

	template< class input_iterator, class container >
	bs_array_shared(
			input_iterator start,
			input_iterator finish,
			const Loki::Type2Type< container >& t = def_cont_tag()
			)
	{
		init(t, start, finish);
	}

	// more convinient factory functions - no need to use Loki::Type2Type
	template< class container >
	static sp_array_shared create(size_type n = 0, const value_type& v = value_type()) {
		return new bs_array_shared(n, v, Loki::Type2Type< container >());
	}

	// more convinient factory functions - no need to use Loki::Type2Type
	template< class container, class input_iterator >
	static sp_array_shared create(input_iterator start, input_iterator finish) {
		return new bs_array_shared(start, finish, Loki::Type2Type< container >());
	}

	//template< class container, class input_iterator >
	//bs_array_shared(input_iterator start, input_iterator finish) {
	//	init< container >(start, finish);
	//}

	// array doesn't own data
	bs_array_shared(pointer data, size_type n)
		: buf_holder_(NULL), data_(data), size_(n)
	{}

	// can be called in any time to switch container
	virtual void init_inplace(const container& c) {
		if((buf_holder_ = c)) {
			if((size_ = buf_holder_->size()))
				data_ = &buf_holder_->ss(0);
		}
	}

	bs_array_shared(const container& c) {
		init_inplace(c);
	}

	// std copy ctor is fine and make a reference to data_

	iterator begin() {
		return data_;
	}
	iterator end() {
		return data_ + size_;
	}

	const_iterator begin() const {
		return data_;
	}
	const_iterator end() const {
		return data_ + size_;
	}

	reverse_iterator rbegin() {
		return reverse_iterator(end());
	}
	reverse_iterator rend() {
		return reverse_iterator(begin());
	}

	const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}
	const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}

	size_type size() const {
		return size_;
	}

	reference operator[](const size_type& n) {
		return data_[n];
	}
	const_reference operator[](const size_type& n) const {
		return data_[n];
	}

	bs_array_shared& operator=(const bs_array_shared& rhs) {
		data_ = rhs.data_;
		size_ = rhs.size_;
		buf_holder_ = rhs.buf_holder_;
		return *this;
	}

	reference front() {
		return data_[0];
	}
	const_reference front() const {
		return data_[0];
	}

	reference back() {
		return data_[size_ - 1];
	}
	const_reference back() const {
		return data_[size_ - 1];
	}

	void swap(bs_array_shared& rhs) {
		std::swap(data_, rhs.data_);
		std::swap(size_, rhs.size_);
		std::swap(buf_holder_, rhs.buf_holder_);
	}

	void resize(size_type n) {
		if(buf_holder_ && n != size_) {
			buf_holder_->resize(n);
			if((size_ = buf_holder_->size()))
				data_ = &buf_holder_->ss(0);
			else
				data_ = NULL;
			//return true;
		}
		//return false;
	}

	friend bool operator==(const bs_array_shared& lhs, const bs_array_shared& rhs) {
		return lhs.data_ == rhs.data_;
	}

	friend bool operator<(const bs_array_shared& lhs, const bs_array_shared& rhs) {
		return lhs.data_ < rhs.data_;
	}

	// explicitly make array copy
	sp_arrbase clone() const {
		return create< def_cont_impl >(begin(), end());
	}

	void dispose() const {
		delete this;
	}

	// access to container
	container get_container() const {
		return buf_holder_;
	}

	//void assign(const value_type& v) {
	//	std::fill(begin(), end(), v);
	//}

protected:
	// if shared array owns buffer here's real buffer handler
	// otherwise NULL
	container buf_holder_;
	// pointer to raw data
	pointer data_;
	size_type size_;
};

}   // eof blue_sky

#endif /* end of include guard: BS_ARRAY_SHARED_529LXCQA */
