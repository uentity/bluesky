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

#include "shared_array.h"

namespace blue_sky {

// bs_array_shared = shared_array + some nessessary ctors and methods
template< class T >
class bs_array_shared : public private_::shared_array< T > {
public:
	typedef private_::shared_array< T > base_t;
	typedef bs_array_shared< T > this_t;

	typedef typename base_t::value_type value_type;
	typedef typename base_t::reference reference;
	typedef typename base_t::const_reference const_reference;
	typedef typename std::allocator <T>::pointer pointer;
	typedef typename base_t::size_type size_type;

	typedef typename base_t::iterator iterator;
	typedef typename base_t::const_iterator const_iterator;

	using base_t::allocator_;

	using base_t::begin;
	using base_t::end;

	// ctors
	bs_array_shared() {}
	// copy ctor
	bs_array_shared(const bs_array_shared& a)
		: base_t(a)
	{}

	bs_array_shared(size_type n, const value_type& v = value_type()) {
		//const size_type new_capacity = 1 + std::max(size_type(1), n);
		if(n) {
			pointer new_memory = allocator_.allocate (n);
			std::fill_n(new_memory, n, v);

			this->array_      = new_memory;
			this->array_end_  = new_memory + n;
			this->capacity_   = n;
			//this->capacity_   = new_capacity;

			BS_ASSERT (this->owner_list_->size () == 1) (this->owner_list_->size ());
			change_owner (new_memory, new_memory + n, n);
		}
	}

	// save reference to passed data pointer, but not own it
	// means that data won't be freed!
	bs_array_shared(size_type n, pointer data) {
		// TODO: is that right? -- seems ok
		this->capacity_   = n;
		if(n) {
			this->array_      = data;
			this->array_end_  = data + n;
		}
	}

	template< class input_iterator >
	bs_array_shared(input_iterator first, input_iterator last) {
		if (this->is_owner ())
			ctor_dispatch_ (first, last);
		else
			bs_throw_exception ("Error: bs_array_shared doesn't own data");
	}

	void init(size_type n, pointer data) {
		bs_array_shared(n, data).swap(*this);
	}

	void resize(size_type new_size, const value_type& v = value_type()) {
		if (this->is_owner ()) {
			if (new_size < this->size())
				erase_at_end_ (this->size () - new_size);
			else
				insert_fill_ (this->end(), new_size - this->size(), v);
		}
		else
			bs_throw_exception ("Error: bs_array_shared doesn't own data");
	}

	void clear () {
		if (this->is_owner ())
			erase_at_end_ (this->size ());
		else
			bs_throw_exception ("Error: bs_array_shared doesn't own data");
	}

	void swap(bs_array_shared& arr) {
		base_t::swap(arr);
	}

private:
	template <typename forward_iterator>
	void ctor_dispatch_ (forward_iterator first, forward_iterator last) {
		const size_type n = std::distance (first, last);
		//size_type new_capacity = 1 + std::max(1, n);
		if(n) {
			pointer new_memory = allocator_.allocate (n);
			pointer new_finish = std::copy(first, last, new_memory);

			this->array_      = new_memory;
			this->array_end_  = new_finish;
			this->capacity_   = n;

			BS_ASSERT (this->owner_list_->size () == 1) (this->owner_list_->size ());
			change_owner (new_memory, new_finish, n);
		}
	}

	void erase_at_end_ (size_type n) {
		destroy (this->end () - n, this->end ());
		this->change_owner (this->end () - n);
	}

	template < typename forward_iterator >
	void destroy(forward_iterator first, forward_iterator last) {
		for (; first != last; ++first)
			allocator_.destroy (&*first);
	}

	void insert_fill_(iterator pos, size_type n, const value_type &value) {
		if(!n) return;
		if ((this->capacity () - this->size ()) >= n) {
			size_type elems_after = this->end () - pos;
			pointer old_finish = this->array_end_;

			if (elems_after > n) {
				std::copy(old_finish - n, old_finish, old_finish);
				std::copy_backward (pos, old_finish - n, old_finish);
				std::fill (pos, pos + n, value);

				this->change_owner (this->end () + n);
			}
			else {
				std::fill_n(old_finish, n - elems_after, value);
				this->array_end_ += n - elems_after;
				std::copy(pos, old_finish, this->end ());
				this->array_end_ += elems_after;
				std::fill (pos, old_finish, value);

				this->change_owner (this->end ());
			}
		}
		else {
			BS_ASSERT (this->is_owner ());
			const size_type new_capacity = std::max(this->size(), n);
			pointer new_memory = allocator_.allocate (new_capacity);
			pointer new_finish = new_memory;

			new_finish = insert_fill_copy_a(pos, n, value, new_memory);

			destroy (this->begin (), this->end ());
			// deallocate
			if(this->begin())
				allocator_.deallocate(this->begin(), this->capacity());
			this->change_owner (new_memory, new_finish, new_capacity);
		}
	}

	T* insert_fill_copy_a (
			T *pos, size_type n, const T &value,
			T *new_memory)
	{
		T *new_finish = std::copy(begin (), pos, new_memory);
		std::fill_n(new_finish, n, value);
		new_finish += n;
		new_finish = std::copy(pos, end(), new_finish);

		return new_finish;
	}
};

} 	// eof blue_sky

#endif /* end of include guard: BS_ARRAY_SHARED_529LXCQA */
