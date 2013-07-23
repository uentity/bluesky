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
	typedef bs_array_shared< T >         bs_array_base;

	typedef smart_ptr< bs_array_base, true > sp_array_shared;
	typedef typename arrbase::sp_arrbase     sp_arrbase;
	typedef bs_array< T, vector_traits >     def_cont_impl;
	//typedef bs_arrbase_impl< T, std::vector< T > > def_cont_impl;
	typedef Loki::Type2Type< def_cont_impl > def_cont_tag;

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

	template< class container_t >
	void init(Loki::Type2Type< container_t >, size_type n = 0, const value_type& v = value_type()) {
		buf_holder_ = NULL;
		if(smart_ptr< container_t > c = create_buf< container_t >()) {
			if(n)
				c->init(n, v);
			buf_holder_ = c;
		}
	}

	template< class input_iterator, class container_t >
	void init(Loki::Type2Type< container_t >, input_iterator start, input_iterator finish) {
		buf_holder_ = NULL;
		//smart_ptr< container_t > c = BS_KERNEL.create_object(container_t::bs_type());
		if(smart_ptr< container_t > c = create_buf< container_t >()) {
			c->init(start, finish);
			buf_holder_ = c;
		}
	}

	// ctors, array ownes data
	bs_array_shared(size_type n = 0, const value_type& v = value_type()) {
		init(def_cont_tag(), n, v);
	}

	template< class container_t >
	bs_array_shared(
			size_type n = 0,
			const value_type& v = value_type(),
			const Loki::Type2Type< container_t >& t = def_cont_tag()
			)
	{
		init(t, n, v);
	}

	template< class input_iterator, class container_t >
	bs_array_shared(
			input_iterator start,
			input_iterator finish,
			const Loki::Type2Type< container_t >& t = def_cont_tag()
			)
	{
		init(t, start, finish);
	}

	// more convinient factory functions - no need to use Loki::Type2Type
	template< class container_t >
	static sp_array_shared create(size_type n = 0, const value_type& v = value_type()) {
		return new bs_array_shared(n, v, Loki::Type2Type< container_t >());
	}

	template< class container_t, class input_iterator >
	static sp_array_shared create(input_iterator start, input_iterator finish) {
		return new bs_array_shared(start, finish, Loki::Type2Type< container_t >());
	}

	// can be called in any time to switch container
	virtual void init_inplace(const container& c) {
		if(buf_holder_.get() != c.get())
			buf_holder_ = c;
	}

	bs_array_shared(const container& c) {
		init_inplace(c);
	}

	static sp_array_shared create(const container& c) {
		return new bs_array_shared(c);
	}

	// std copy ctor is fine and make a reference to data_

	// implement bs_arrbase interface
	size_type size() const {
		if(buf_holder_)
			return buf_holder_->size();
		else
			return 0;
	}

	void resize(size_type n) {
		if(buf_holder_ && size() != n)
			buf_holder_->resize(n);
	}

	pointer data() {
		if(buf_holder_)
			return buf_holder_.lock()->data();
		else
			return NULL;
	}

	const_pointer data() const {
		if(buf_holder_)
			return buf_holder_->data();
		else
			return NULL;
	}

	// explicitly make array copy
	sp_arrbase clone() const {
		return create< def_cont_impl >(this->begin(), this->end());
	}

	// reference to rhs array
	bs_array_shared& operator=(const bs_array_shared& rhs) {
		buf_holder_ = rhs.buf_holder_;
		return *this;
	}

	void swap(bs_array_shared& rhs) {
		std::swap(buf_holder_, rhs.buf_holder_);
	}

	friend bool operator==(const bs_array_shared& lhs, const bs_array_shared& rhs) {
		return lhs.data() == rhs.data();
	}

	friend bool operator<(const bs_array_shared& lhs, const bs_array_shared& rhs) {
		return lhs.data() < rhs.data();
	}

	void dispose() const {
		delete this;
	}

	// access to container
	container get_container() const {
		return buf_holder_;
	}

protected:
	// if shared array owns buffer here's real buffer handler
	// otherwise NULL
	container buf_holder_;

	// helpers to create BS objects via kernel
	// and others via new
	template< class container_t >
	static smart_ptr< container_t > create_buf_(Loki::Int2Type< 0 >) {
		return new container_t;
	}
	template< class container_t >
	static smart_ptr< container_t > create_buf_(Loki::Int2Type< 1 >) {
		return BS_KERNEL.create_object(container_t::bs_type());
	}

	// main container creator
	template< class container_t >
	static smart_ptr< container_t > create_buf() {
		return create_buf_< container_t >(
			Loki::Int2Type< conversion< container_t, objbase >::exists_uc >()
		);
	}
};

}   // eof blue_sky

#endif /* end of include guard: BS_ARRAY_SHARED_529LXCQA */
