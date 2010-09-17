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
#ifndef BS_VECTOR_SHARED_8W625HOK
#define BS_VECTOR_SHARED_8W625HOK

#include "bs_array_shared.h"

namespace blue_sky {

/*-----------------------------------------------------------------
 * Make array around plain buffer optionally held by bs_arrbase-compatible container 
 *----------------------------------------------------------------*/
template< class T >
class BS_API bs_vector_shared : public bs_array_shared< T >, public bs_vecbase< T > {
public:
	typedef bs_array_shared< T > base_t;
	// traits for bs_array
	typedef bs_arrbase< T > arrbase;
	typedef typename base_t::container container;
	typedef bs_vector_shared< T > bs_array_base;

	typedef smart_ptr< bs_array_base, true > sp_vector_shared;
	typedef typename arrbase::sp_arrbase sp_arrbase;
	typedef typename base_t::def_cont_impl def_cont_impl;
	typedef typename base_t::def_cont_tag def_cont_tag;
	typedef bs_vecbase< T > vecbase_t;

	typedef typename base_t::value_type value_type;
	typedef typename base_t::size_type size_type;
	typedef typename base_t::iterator iterator;
	typedef typename vecbase_t::key_type key_type;

	// ctors, array ownes data
	bs_vector_shared(size_type n = 0, const value_type& v = value_type())
		: base_t(n, v, def_cont_tag())
	{
		assert(buf_holder_.get());
		pvec_ = const_cast< vecbase_t* >(static_cast< const vecbase_t* >((const def_cont_impl*)buf_holder_.get()));
	}

	template< class container >
	bs_vector_shared(
		size_type n = 0,
		const value_type& v = value_type(),
		const Loki::Type2Type< container >& t = def_cont_tag()
		)
		: base_t(n, v, t)
	{
		assert(buf_holder_.get());
		pvec_ = const_cast< vecbase_t* >(static_cast< const vecbase_t* >((const container*)buf_holder_.get()));
	}

	template< class input_iterator, class container >
	bs_vector_shared(
			input_iterator start,
			input_iterator finish,
			const Loki::Type2Type< container >& t = def_cont_tag()
			)
		: base_t(start, finish, t)
	{
		assert(buf_holder_.get());
		pvec_ = const_cast< vecbase_t* >(static_cast< const vecbase_t* >((const container*)buf_holder_.get()));
	}

	// more convinient factory functions - no need to use Loki::Type2Type
	template< class container >
	static sp_vector_shared create(size_type n = 0, const value_type& v = value_type()) {
		return new bs_vector_shared(n, v, Loki::Type2Type< container >());
	}

	// more convinient factory functions - no need to use Loki::Type2Type
	template< class container, class input_iterator >
	static sp_vector_shared create(input_iterator start, input_iterator finish) {
		return new bs_vector_shared(start, finish, Loki::Type2Type< container >());
	}

	// ctor with external data
	// use dynamic cast to obtain bs_vecbase iface
	bs_vector_shared(const container& c)
		: base_t(c)
	{
		assert(buf_holder_);
		pvec_ = const_cast< vecbase_t* >(static_cast< const vecbase_t* >((const container*)buf_holder_.get()));
		BS_ERROR(pvec_, "Container passed to bs_vector_shared ctor doesn't have vector iface");
	}

	void push_back(const value_type& v) {
		vecbuf()->push_back(v);
	}

	void pop_back() {
		vecbuf()->pop_back();
	}

	iterator insert(iterator pos, const value_type& v) {
		return vecbuf()->insert(pos, v);
	}
	void insert(iterator pos, size_type n, const value_type& v) {
		vecbuf()->insert(pos, n, v);
	}

	//template< class input_iterator >
	//void insert(iterator pos, input_iterator start, input_iterator finish) {
	//	vecbuf()->insert(pos, start, finish);
	//}

	iterator erase(iterator pos) {
		return vecbuf()->erase(pos);
	}
	iterator erase(iterator start, iterator finish) {
		return vecbuf()->erase(start, finish);
	}

	// overloads from bs_vecbase
	bool insert(const key_type& key, const value_type& value) {
		return vecbuf()->insert(key, value);
	}

	bool insert(const value_type& value) {
		return vecbuf()->insert(value);
	}

	void erase(const key_type& key)	{
		vecbuf()->erase(key);
	}

	void clear() {
		vecbuf()->clear();
	}

	// explicitly make array copy
	sp_arrbase clone() const {
		return create< def_cont_impl >(this->begin(), this->end());
	}

private:
	using base_t::buf_holder_;
	using base_t::size_;
	using base_t::data_;

	// we can use dynamic_cast or store pointer to bs_vecbase iface
	vecbase_t* pvec_;

	// RAII for updating data_ and size_ after eeach operation
	struct vb_handle {
		explicit vb_handle(bs_vector_shared& v) : self_(v) {}

		~vb_handle() {
			self_.upd_data();
		}

		inline vecbase_t* operator->() {
			return self_.pvec_;
			//return const_cast< vecbase_t* >(dynamic_cast< vecbase_t const* >(self_.buf_holder_.get()));
		}

		bs_vector_shared& self_;
	};
	// static cast to bs_vecbase
	inline vb_handle vecbuf() {
		return vb_handle(*this);
	}
	//inline vecbase_t const* vecbuf() const {
	//	return static_cast< const vecbase_t* >(buf_holder_.get());
	//}

	inline void upd_data() {
		size_ = buf_holder_->size();
		if(size_)
			data_ = &buf_holder_->ss(0);
		else
			data_ = NULL;
	}
};

} 	// eof blue_sky

#endif /* end of include guard: BS_VECTOR_SHARED_8W625HOK */

