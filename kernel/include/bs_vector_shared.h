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
	// ensure we create a vector container
	typedef bs_array< T, vector_traits > def_cont_impl;
	typedef Loki::Type2Type< def_cont_impl > def_cont_tag;
	//typedef typename base_t::def_cont_impl def_cont_impl;
	//typedef typename base_t::def_cont_tag def_cont_tag;
	typedef bs_vecbase< T > vecbase_t;

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

	// ctors, array ownes data
	bs_vector_shared(size_type n = 0, const value_type& v = value_type())
		: base_t(n, v, def_cont_tag())
	{
		pvec_ = container2vecbase();
	}

	template< class container_t >
	bs_vector_shared(
		size_type n = 0,
		const value_type& v = value_type(),
		const Loki::Type2Type< container_t >& t = def_cont_tag()
		)
		: base_t(n, v, t)
	{
		pvec_ = container2vecbase();
	}

	template< class input_iterator, class container_t >
	bs_vector_shared(
			input_iterator start,
			input_iterator finish,
			const Loki::Type2Type< container_t >& t = def_cont_tag()
			)
		: base_t(start, finish, t)
	{
		pvec_ = container2vecbase();
	}

	// more convinient factory functions - no need to use Loki::Type2Type
	template< class container_t >
	static sp_vector_shared create(size_type n = 0, const value_type& v = value_type()) {
		return new bs_vector_shared(n, v, Loki::Type2Type< container_t >());
	}

	// more convinient factory functions - no need to use Loki::Type2Type
	template< class container_t, class input_iterator >
	static sp_vector_shared create(input_iterator start, input_iterator finish) {
		return new bs_vector_shared(start, finish, Loki::Type2Type< container_t >());
	}

	// ctor with external data
	// use dynamic cast to obtain bs_vecbase iface
	bs_vector_shared(const container& c)
		: base_t(c)
	{
		init_inplace(c);
	}

	void init_inplace(const container& c) {
		if(!c.get()) return;
		// try to dynamically cast from c to vecbase
		pvec_ = const_cast< vecbase_t* >(dynamic_cast< const vecbase_t* >(c.get()));
		if(pvec_)
			base_t::init_inplace(c);
		else {
			// if container is not vector-based then make a copy of data
			base_t::init(def_cont_tag(), c->begin(), c->end());
			pvec_ = container2vecbase();
		}
	}

	void push_back(const value_type& v) {
		pvec_->push_back(v);
	}

	void pop_back() {
		pvec_->pop_back();
	}

	iterator insert(iterator pos, const value_type& v) {
		return pvec_->insert(pos, v);
	}
	void insert(iterator pos, size_type n, const value_type& v) {
		pvec_->insert(pos, n, v);
	}

	//template< class input_iterator >
	//void insert(iterator pos, input_iterator start, input_iterator finish) {
	//	pvec_->insert(pos, start, finish);
	//}

	iterator erase(iterator pos) {
		return pvec_->erase(pos);
	}
	iterator erase(iterator start, iterator finish) {
		return pvec_->erase(start, finish);
	}

	// overloads from bs_vecbase
	bool insert(const key_type& key, const value_type& value) {
		return pvec_->insert(key, value);
	}

	bool insert(const value_type& value) {
		return pvec_->insert(value);
	}

	void erase(const key_type& key)	{
		pvec_->erase(key);
	}

	void clear() {
		pvec_->clear();
	}

	void reserve(size_type sz) {
		pvec_->reserve(sz);
	}

	// explicitly make array copy
	sp_arrbase clone() const {
		return create< def_cont_impl >(this->begin(), this->end());
	}

private:
	using base_t::buf_holder_;
	// we can use dynamic_cast or store pointer to bs_vecbase iface
	vecbase_t* pvec_;

	inline vecbase_t* container2vecbase() const {
		if(buf_holder_.get())
			return const_cast< vecbase_t* >(
				static_cast< const vecbase_t* >((const def_cont_impl*)buf_holder_.get())
			);
		else
			return NULL;
	}
};

} 	// eof blue_sky

#endif /* end of include guard: BS_VECTOR_SHARED_8W625HOK */

