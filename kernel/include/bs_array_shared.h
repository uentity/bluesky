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

#include "bs_common.h"

namespace blue_sky {

// wrap array around smart_ptr< std::vector< T > >
template< class container_t >
class BS_API bs_array_shared {
public:
	typedef container_t container;
	typedef st_smart_ptr< container > sp_container;

	typedef typename container::value_type             value_type;
	typedef typename container::pointer                pointer;
	typedef typename container::const_pointer          const_pointer;
	typedef typename container::reference              reference;
	typedef typename container::const_reference        const_reference;
	typedef typename container::iterator               iterator;
	typedef typename container::const_iterator         const_iterator;
	typedef typename container::reverse_iterator       reverse_iterator;
	typedef typename container::const_reverse_iterator const_reverse_iterator;
	typedef typename container::size_type              size_type;
	typedef typename container::difference_type        difference_type;
	typedef typename container::allocator_type         allocator_type;

	// ctora
	bs_array_shared()
		: data_(new container)
	{}

	bs_array_shared(size_type n, const value_type& v = value_type())
		: data_(new container(n, v))
	{}

	template< class input_iterator >
	bs_array_shared(input_iterator start, input_iterator finish)
		: data_(new container(start, finish))
	{}

	// std copy ctor is fine and make a reference to data_

	iterator begin() {
		return data_->begin();
	}
	iterator end() {
		return data_->end();
	}

	const_iterator begin() const {
		return data_->begin();
	}
	const_iterator end() const {
		return data_->end();
	}

	reverse_iterator rbegin() {
		return data_->rbegin();
	}
	reverse_iterator rend() {
		return data_->rend();
	}

	const_reverse_iterator rbegin() const {
		return data_->rbegin();
	}
	const_reverse_iterator rend() const {
		return data_->rend();
	}

	size_type size() const {
		return data_->size();
	}

	reference operator[](size_type n) {
		return data_->operator[](n);
	}
	const_reference operator[](size_type n) const {
		return data_->operator[](n);
	}

	bs_array_shared& operator=(const bs_array_shared& rhs) {
		*data_ = *rhs.data_;
		return *this;
	}

	reference front() {
		return data_->front();
	}
	const_reference front() const {
		return data_->front();
	}

	reference back() {
		return data_->back();
	}
	const_reference back() const {
		return data_->back();
	}

	void swap(bs_array_shared& rhs) {
		std::swap(data_, rhs.data_);
	}

	void resize(size_type n, const value_type& v = value_type()) {
		data_->resize(n, v);
	}

	template< class r_container >
	friend bool operator==(const bs_array_shared& lhs, const bs_array_shared< r_container >& rhs) {
		return *lhs.data_ == *rhs.data_;
	}

	template< class r_container >
	friend bool operator<(const bs_array_shared& lhs, const bs_array_shared< r_container >& rhs) {
		return *lhs.data_ < *rhs.data_;
	}

	// explicitly make array copy
	bs_array_shared clone() const {
		bs_array_shared t(begin(), end());
		return t;
	}

protected:
	sp_container data_;
};

template< class container_t >
class BS_API bs_vector_shared : public bs_array_shared< container_t > {
public:
	typedef bs_array_shared< container_t > base_t;

	typedef typename base_t::value_type value_type;
	typedef typename base_t::size_type size_type;
	typedef typename base_t::iterator iterator;
	using base_t::data_;
	using base_t::begin;
	using base_t::end;

	// ctora
	bs_vector_shared() {}

	bs_vector_shared(size_type n, const value_type& v = value_type())
		: base_t(n, v)
	{}

	template< class input_iterator >
	bs_vector_shared(input_iterator start, input_iterator finish)
		: base_t(start, finish)
	{}

	void push_back(const value_type& v) {
		data_->push_back(v);
	}

	void pop_back(const value_type& v) {
		data_->pop_back(v);
	}

	iterator insert(iterator pos, const value_type& v) {
		return data_->insert(pos, v);
	}
	void insert(iterator pos, size_type n, const value_type& v) {
		data_->insert(pos, n, v);
	}

	template< class input_iterator >
	void insert(iterator pos, input_iterator start, input_iterator finish) {
		data_->insert(pos, start, finish);
	}

	iterator erase(iterator pos) {
		return data_->erase(pos);
	}
	iterator erase(iterator start, iterator finish) {
		return data_->erase(start, finish);
	}

	void clear() {
		data_->clear();
	}

	// explicitly make array copy
	bs_vector_shared clone() const {
		bs_vector_shared t(begin(), end());
		return t;
	}
};

} 	// eof blue_sky

#endif /* end of include guard: BS_ARRAY_SHARED_529LXCQA */
