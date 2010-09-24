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

#include "bs_tree.h"
#include "bs_command.h"
#include "bs_exception.h"
#include "bs_prop_base.h"
#include "bs_kernel.h"
#include "bs_conversion.h"

#include <set>
#include <map>
#include <list>
#include <algorithm>
#include <sstream>

#include "boost/pool/pool_alloc.hpp"
#include "boost/detail/atomic_count.hpp"

#include "loki/TypeManip.h"

//using namespace blue_sky;
using namespace boost;
using namespace std;
using namespace Loki;

namespace blue_sky {
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (bs_node::s_traits_ptr, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (sp_link, str_val_traits));

namespace {
//hide implementation

//=============================== helper structures implementation =====================================================
//link comparision by name
struct lcomp_by_name {
	bool operator()(const sp_link& lhs, const sp_link& rhs) const {
		return lhs->name() < rhs->name();
	}
};

//link comparision by inode pointer
struct lcomp_by_ptr {
	bool operator()(const sp_link& lhs, const sp_link& rhs) const {
		return lhs->inode() < rhs->inode();
	}
};

template< class T >
struct elem_ptr {
	typedef T elem_t;
	//nil ctor
	elem_ptr() : p_(&nil_el) {}

	elem_ptr(const T& el) : p_(const_cast< T* >(&el)) {}

	elem_ptr& operator =(const T& el) {
		p_ = const_cast< T* >(&el);
		return *this;
	}

	elem_ptr& operator =(const T* p_el) {
		p_ = const_cast< T* >(p_el);
		return *this;
	}

	T* operator ->() const { return p_; }

	//conversion to elem_t
	T& elem() const { return *p_; }
	T& operator *() const { return *p_; }

	bool is_nil() const {
		return p_ == &nil_el;
	}

	operator bool() const {
		return !is_nil();
	}

	bool operator==(const elem_ptr& lhs) const {
		return p_ == lhs.p_;
	}

	bool operator!=(const elem_ptr& lhs) const {
		return p_ != lhs.p_;
	}

	bool operator<(const elem_ptr& lhs) const {
		return p_ < lhs.p_;
	}	

	static T nil_el;

private:
	T* p_;
};

struct bs_leaf;

//---------------------------------- key used by custom index ----------------------------------------------------------
struct node_key {
	friend struct bs_leaf;

public:
	enum key_state {
		key_ok = 0,
		key_type_conflict,
		key_bad
	};

	node_key(const sp_link& l, const bs_node::s_traits_ptr& s) {
		//if l is NULL - return empty key
		if(!l || !s) return;

		if(s->accepts(l))
			k_ = s->key_generator(l);
		else {
			string msg = "Link """;
			msg += l->name() + """ is incompatible with sorting " + s->sort_name();
			throw bs_exception("BlueSky kernel", msg.c_str());
		}
	}

	//public ctors - used for searching
	node_key(const bs_node::sort_traits::key_ptr& k) {
		k_ = k;
	}

	bool operator < (const node_key& nk) const {
		if(k_ && nk.k_ && typeid(*k_) == typeid(*nk.k_))
			return k_->sort_order(nk.k_);

		return false;
	}

	key_state update_key(const sp_link& l, const bs_node::s_traits_ptr& s) {
		try {
			node_key tmp(l, s);
			if(tmp) tmp.swap(*this);
			else return key_bad;
		}
		catch(const bs_exception&) {
			return key_type_conflict;
		}
		return key_ok;
	}

	bool is_empty() const {
		return (k_.get() == NULL);
	}

	operator bool() const {
		return !is_empty();
	}

	void clear() {
		k_ = NULL;
	}

private:
	bs_node::sort_traits::key_ptr k_;

	void swap(node_key& k) {
		std::swap(k_, k.k_);
//		bs_node::sort_traits::key_ptr tmp = k_;
//		k_ = k.k_;
//		k.k_ = tmp;
	}

	//builds empty key
	node_key() {};
};

//-------------------------- leaf tuple: key + link contained in inodep & custom indexes -------------------------------
struct bs_leaf {
	typedef elem_ptr< bs_leaf > leaf_ptr;

	node_key key_;
	sp_link link_;
	// persistence flag
	bool is_persistent_;

	bs_leaf(const node_key& k, const sp_link& l = NULL, bool is_persistent = false)
		: key_(k), link_(l), is_persistent_(is_persistent)
	{}

	//full initialization
	bs_leaf(const sp_link& l, const bs_node::s_traits_ptr& s, bool is_persistent)
		: key_(node_key(l, s)), link_(l), is_persistent_(is_persistent)
	{}

	//construct with null key only for searching by ptr
	bs_leaf(const sp_link& l, bool is_persistent = false) : key_(), link_(l), is_persistent_(is_persistent) {}

	// behave like leaf_ptr (allow unified syntax)
	const bs_leaf* operator->() const {
		return this;
	}

	bs_leaf* operator->() {
		return this;
	}

	//void set_link(const sp_link& l) {
	//	if(l) link_ = *l;
	//	else link_ = nil_link_s;
	//}

	node_key::key_state update_key(const bs_node::s_traits_ptr& s) {
		return key_.update_key(link_, s);
	}

	void clear_key() {
		key_.clear();
	}

	// persistense flag access
	void set_persistence(bool persistent) {
		// block using sp_link's mutex
		bs_mutex::scoped_lock guard(*link_.mutex());
		is_persistent_ = persistent;
	}
	bool is_persistent() const { return is_persistent_; }

	//leafs comparison predicates
	//by link name
	struct comp_by_name {
		bool operator()(const bs_leaf& lhs, const bs_leaf& rhs) const {
			//if(lhs.link_ && rhs.link_)
			return (lhs.link_->name() < rhs.link_->name());
		}
	};

	//by key - for sorting
	struct comp_by_key {
		bool operator()(const leaf_ptr& lhs, const leaf_ptr& rhs) const {
			return (lhs->key_ < rhs->key_);
		}
	};

	//by embed bs_inode pointer
	struct comp_by_ptr {
		bool operator()(const bs_leaf& lhs, const bs_leaf& rhs) const {
			return (lhs.link_->inode() < rhs.link_->inode());
		}
	};
};
//pointers to leafs will be used in order container
typedef bs_leaf::leaf_ptr leaf_ptr;
//nil element definition for leaf_ptr
template< > bs_leaf leaf_ptr::nil_el = bs_leaf(NULL);

//------------------------------ typedefs used in node_impl class ------------------------------------------------------
//fast allocator for sp_link pointers
typedef fast_pool_allocator< sp_link, default_user_allocator_new_delete,
	details::pool::null_mutex > sp_link_alloc;

//leafs allocator
typedef fast_pool_allocator< bs_leaf, default_user_allocator_new_delete,
	details::pool::null_mutex > leaf_alloc;

typedef fast_pool_allocator< leaf_ptr, default_user_allocator_new_delete,
	details::pool::null_mutex > leaf_ptr_alloc;

//raw links container sorted by unique (among this node) link name
typedef set< sp_link, lcomp_by_name, sp_link_alloc > name_index_t;
//typedef Int2Type< bs_node::name_idx > name_idx_tag;

//order of leafs visit determined by order of node_key's
typedef multiset< leaf_ptr, bs_leaf::comp_by_key, leaf_ptr_alloc > cust_index_t;
//typedef Int2Type< bs_node::custom_idx > cust_idx_tag;

//raw links container sorted by bs_inode pointers
typedef multiset< bs_leaf, bs_leaf::comp_by_ptr, leaf_alloc > inodep_index_t;
//typedef Int2Type< bs_node::inodep_idx > ip_idx_tag;

//---------------------------- index type to tag binding ---------------------------------------------------------------
template< class index_t >
struct idx_traits {};

template< >
struct idx_traits< name_index_t > {
	typedef name_index_t type;
	typedef type::value_type val_t;
	typedef type::key_type key_t;
	enum { tag = bs_node::name_idx };
};

template< >
struct idx_traits< cust_index_t > {
	typedef cust_index_t type;
	typedef type::value_type val_t;
	typedef type::key_type key_t;
	enum { tag = bs_node::custom_idx };
};

template< >
struct idx_traits< inodep_index_t > {
	typedef inodep_index_t type;
	typedef type::value_type val_t;
	typedef type::key_type key_t;
	enum { tag = bs_node::inodep_idx };
};

typedef idx_traits< name_index_t > name_idx_tag;
typedef idx_traits< cust_index_t > cust_idx_tag;
typedef idx_traits< inodep_index_t > ip_idx_tag;

template< bs_node::index_type idx_id >
struct idx_tag2type {
private:
	template< bs_node::index_type idx_t >
	struct select_idx_traits {
		enum { is_name_idx = idx_traits< name_index_t >::tag == (int)idx_t };
		//enum { is_cust_idx = typename idx_traits< cust_index_t >::tag == idx_t };
		enum { is_inodep_idx = idx_traits< inodep_index_t >::tag == (int)idx_t };

		typedef typename Select< is_name_idx, name_index_t, cust_index_t >::Result name_or_cust;
		typedef typename Select< is_inodep_idx, inodep_index_t, name_or_cust >::Result res_idx_t;
		typedef idx_traits< res_idx_t > res_idx_traits;
	};

public:
	typedef typename select_idx_traits< idx_id >::res_idx_t type;
	typedef idx_traits< type > traits;
};

//---------------- iter2link helper function
template< class i_traits >
const sp_link& iter2link(const typename i_traits::type::const_iterator& i, i_traits) { return (*i)->link_; }

template< >
const sp_link& iter2link< name_idx_tag >(const name_idx_tag::type::const_iterator& i, name_idx_tag)
{ return *i; }

} //end of unnamed hidden namespace


//=============================== node implementation ==================================================================
class bs_node::node_impl {
	friend struct node_key;
	friend struct bs_leaf;

public:
	typedef smart_ptr< node_impl, false > sp_nimpl;

	//---------------- link2key helpers
	inline sp_link link2key(const sp_link& l, name_idx_tag) const { return l; }
	inline bs_leaf link2key(const sp_link& l, ip_idx_tag) const { return bs_leaf(l); }
	inline bs_leaf link2key(const sp_link& l, cust_idx_tag) const { return bs_leaf(l, sort_, false); }
	//---------------- and reverse conversion
	inline sp_link key2link(const sp_link& l, name_idx_tag) const { return l; }
	inline sp_link key2link(const bs_leaf& leaf, ip_idx_tag) const { return leaf.link_; }
	inline sp_link key2link(const leaf_ptr& leaf, cust_idx_tag) const { return leaf->link_; }

	//the same, but for searching - meaningful only for custom index
	inline sp_link link2srch_key(const sp_link& l, name_idx_tag) const { return l; }
	inline bs_leaf link2srch_key(const sp_link& l, ip_idx_tag) const { return bs_leaf(l); }
	inline bs_leaf link2srch_key(const sp_link& l, cust_idx_tag) const { return bs_leaf(l); }

	//----------------- slot to handle leafs updates
	class leaf_tracker : public bs_slot {
	public:
		leaf_tracker(node_impl& ni) : ni_(ni) {
			// DEBUG!
			//if(ni_.self_) {
			//	cout << "leaf_tracker bs_node mutex at " << &ni_.self_->mutex() << endl;
			//	cout << "leaf_tracker sp_node mutex at " << sp_node(ni_.self_).mutex() << endl;
			//	cout << "leaf_tracker node_impl mutex at " << ni_.self_->pimpl_.mutex() << endl;
			//}
		}

		void execute(const sp_mobj& sender, int, const sp_obj&) const {
			//sender_ was updated - so we should update custom index accordingly
			sp_link l(sender, bs_dynamic_cast());
			if(!l) return;

			//DEBUG
			//cout << "leaf_tracker called for link " << l->name() << endl;

			//do everything in single transaction
			//block owner node
			lsmart_ptr< sp_node > ni_lock(sp_node(ni_.self_));
			//block customm index
			bs_mutex::scoped_lock lk(ni_.ci_guard_);
			//find all references to given object
			pair< inodep_index_t::iterator, inodep_index_t::iterator > rgn =
				ni_.ip_idx_.equal_range(ni_.link2key(l, ip_idx_tag()));

			//update all found leafs in cycle
			for(inodep_index_t::iterator p = rgn.first; p != rgn.second; ++p) {
				//update found leaf
				ni_.update_custom_idx(const_cast< ip_idx_tag::val_t& >(*p));
			}
			//return NULL;
		}

//		bool can_unexecute() const {
//			return false;
//		}
//
//		void unexecute() {}

	private:
		node_impl& ni_;
		//atomicity guard
		//mutable bs_mutex guard_;
	};

	//------------------------------- member variables ---------------------------------------------
	//pointer to node's master class
	bs_node* self_;

	//order of sorting is stored here
	s_traits_ptr sort_;

	//leafs tracker
	sp_slot lt_;

	//name index
	name_index_t leafs_;

	//custom index for custom sorting
	cust_index_t cust_idx_;
	//guard to specially protect custom index
	bs_mutex ci_guard_;

	//inode ptr index for tracking signals from pointed objects
	inodep_index_t ip_idx_;


	//------------------------------------------node_impl ctors ------------------------------------
	node_impl() : self_(NULL)
	{}

	node_impl(const s_traits_ptr& s) : self_(NULL)
	{
		if(s) sort_ = s;
	}

	//node_impl(const s_traits_stack& st)
	//	: sort_(st.begin(), st.end())
	//{}

	//explicit copy ctor - doesn't copy mutex
	node_impl(const node_impl& ni)
		: self_(NULL), sort_(ni.sort_)
	{
		sp_obj obj_copy;
		for(inodep_index_t::const_iterator i = ni.ip_idx_.begin(), end = ni.ip_idx_.end(); i != end; ++i) {
			const sp_link& cur_l = i->link_;
			obj_copy = NULL;
			if(cur_l->data()->bs_resolve_type().is_copyable())
				obj_copy = give_kernel::Instance().create_object_copy(cur_l->data());
			if(!obj_copy)
				obj_copy = cur_l->data();

			add_link(new bs_link(obj_copy, cur_l->name()), i->is_persistent());
		}
	}

	//self-destructin method
	void dispose() const {
		delete this;
	}

	//-------------------------- static members ----------------------------------------------------
	// global protected counter for creating unique node names
	static boost::detail::atomic_count unique_cnt_s;
	static long next_unique_val() {
		++unique_cnt_s;
		return unique_cnt_s;
	}

	template< class index_t >
	n_iterator si2ni(const typename index_t::const_iterator& i) const;

	template< class index_t, class iter_t >
	bs_node::n_range sr2nr(const pair< iter_t, iter_t >& rng) const;

	//access to different indexes by tag
	inline name_index_t& index(name_idx_tag) { return leafs_; }
	inline const name_index_t& index(name_idx_tag) const { return leafs_; }

	inline inodep_index_t& index(ip_idx_tag) { return ip_idx_; }
	inline const inodep_index_t& index(ip_idx_tag) const { return ip_idx_; }

	inline cust_index_t& index(cust_idx_tag) { return cust_idx_; }
	inline const cust_index_t& index(cust_idx_tag) const { return cust_idx_; }

	// remove constness from index inside const fucntions
	template< class index_tag >
	inline typename index_tag::type& index_uc(index_tag tag) const {
		return const_cast< typename index_tag::type& >(index(tag));
	}

	//---------------------------- count implementation ----------------------------------------------------------------
	ulong count(const sort_traits::key_ptr& k) const {
		return static_cast< ulong >(cust_idx_.count(bs_leaf(node_key(k))));
	}

	ulong count(const sp_obj& obj) const {
		return static_cast< ulong >(ip_idx_.count(bs_leaf(new bs_link(obj, ""))));
	}

	//-------------------------------------- search methods ------------------------------------------------------------
	n_range search(const sort_traits::key_ptr& k) const {
		pair< cust_index_t::const_iterator, cust_index_t::const_iterator > res =
			cust_idx_.equal_range(bs_leaf(node_key(k)));
		return sr2nr< cust_index_t >(res);
	}

	//variation of the above function for only one value (not range)
	//-------------------------------------- peek by name algorithm implementation -------------------------------------
	template< class index_t >
	typename index_t::iterator peek_by_name(const typename index_t::key_type& k, idx_traits< index_t > i_tag) const
	{
		//idx_traits< index_t > i_tag;
		typedef typename index_t::iterator target_iter;
		//remove const
		index_t& uc_idx = index_uc(i_tag);
		//find range of equal leafs
		pair< target_iter, target_iter > rng = uc_idx.equal_range(k);

		//search found region and peek leaf by name
		const sp_link& l = key2link(k, i_tag);
		if(rng.first == rng.second) {
			rng.first = rng.second = uc_idx.end();
		}
		else if(l && !l->name().empty()) {
			const string& link_name = l->name();
			for(; rng.first != rng.second; ++rng.first) {
				if(iter2link(rng.first, i_tag)->name() == link_name)
					return rng.first;
			}
			//leaf with given name wasn't found
			return uc_idx.end();
		}
		return rng.first;
	}

	// peek_by_name that always accepts sp_link as a search key
	template< class index_tag >
	inline typename index_tag::type::iterator peek_by_name(const sp_link& l, index_tag tag) const {
		return peek_by_name< typename index_tag::type >(link2key(l, tag), tag);
	}

	// simplified version of the above for name_index_t
	name_index_t::iterator peek_by_name(const name_index_t::key_type& k, name_idx_tag) const {
		//remove const
		name_index_t& uc_idx = const_cast< name_index_t& >(leafs_);
		//find leaf
		return uc_idx.find(k);
	}

	//---------- raw_find helper
	template< class index_t >
	inline typename index_t::iterator raw_find(const index_t& idx, const sp_link& l) const {
		return const_cast< index_t& >(idx).find(link2key(l, idx_traits< index_t >()));
	}

	//------------------------- search methods implementation --------------------------------------
	//find leaf by name
	name_index_t::iterator find(const string& name, name_idx_tag) const {
		return raw_find(leafs_, bs_link::dumb_link(name));
	}

	template< class index_tag >
	typename index_tag::type::iterator find(const std::string& name, index_tag tag) const {
		name_index_t::iterator pl = find(name, name_idx_tag());
		if(pl != leafs_.end())
			return peek_by_name(*pl, tag);
		else
			return index_uc(tag).end();
	}

	// find leaf by link and possibly match the name
	name_index_t::iterator find(const sp_link& l, name_idx_tag, bool = true) const {
		return raw_find(leafs_, l);
	}

	template< class index_tag >
	typename index_tag::type::iterator find(const sp_link& l, index_tag tag, bool match_name = true) const
	{
		typedef typename index_tag::type index_t;

		index_t& uc_idx = index_uc(tag);
		if(!l)
			//link is incorrect - nothing to search
			return uc_idx.end();

		if(match_name)
			return peek_by_name(l, tag);
		else
			return raw_find(uc_idx, l);
	}

	//------------------------- equal range imlementation --------------------------------------------------------------
	pair< name_index_t::iterator, name_index_t::iterator >
	equal_range(const sp_link& l, name_idx_tag tag) const
	{
		pair< name_index_t::iterator, name_index_t::iterator > rng;
		if(l)
			rng.first = find(l, tag);
		else
			rng.first = index_uc(tag).end();
		rng.second = rng.first;
		if(rng.first != leafs_.end())
			++rng.second;
		return rng;
	}

	template< class index_tag >
	pair< typename index_tag::type::iterator, typename index_tag::type::iterator >
	equal_range(const sp_link& l, index_tag tag) const
	{
		//idx_traits< index_t > tag;
		typedef typename index_tag::type index_t;
		index_t& uc_idx = index_uc(tag);
		if(l)
			return uc_idx.equal_range(link2key(l, tag));
		else {
			typedef typename index_t::iterator iterator;
			pair< iterator, iterator > rng;
			rng.first = uc_idx.end();
			rng.second = rng.first;
			return rng;
		}
	}

	pair< cust_index_t::iterator, cust_index_t::iterator > equal_range(const sort_traits::key_ptr& k) const {
		return index_uc(cust_idx_tag()).equal_range(bs_leaf(node_key(k)));
	}

	//------------------ convert iterator to leaf
	inline leaf_ptr iter2leaf(const name_index_t::const_iterator& i, name_idx_tag) const {
		inodep_index_t::const_iterator p = peek_by_name(*i, ip_idx_tag());
		if(p != ip_idx_.end())
			return *p;
		else return leaf_ptr::nil_el;
	}

	template< class index_t >
	inline leaf_ptr iter2leaf(const typename index_t::const_iterator& i, idx_traits< index_t >) const {
		return *i;
	}

	//------------------ convert leaf to iterator
	inline name_index_t::iterator leaf2iter(const leaf_ptr& leaf, name_idx_tag tag) const {
		return peek_by_name(leaf->link_, tag);
	}

	template< class index_t >
	inline typename index_t::iterator leaf2iter(const leaf_ptr& leaf, idx_traits< index_t > tag) const {
		return peek_by_name((typename index_t::key_type&)leaf, tag);
	}

	//-------------------------------- node_impl leafs operations ------------------------------------------------------
	insert_ret_t add_link(const sp_link& l, bool is_persistent, bool force = false, bool emit_signal = true) {
		pair< name_index_t::iterator, bool > new_l;
		//insert_ret_t res(si2ni< name_index_t >(leafs_.end()), ins_ok);
		insert_ret_t res;
		res.second = ins_ok;

		//if we have sort_traits then first check for type conflict
		if(sort_ && !sort_->accepts(l)) {
			res.first = si2ni< name_index_t >(leafs_.end());
			res.second = ins_type_conflict;
			return res;
		}
		//insert new leaf and check name conflict
		new_l = leafs_.insert(l);
		res.first = si2ni< name_index_t >(new_l.first);
		if(!new_l.second) {
			if(force) {
				//if we are forcing - erase existing link and insert new one
				erase_leaf(new_l.first, name_idx_tag());
				return add_link(l, is_persistent, false, emit_signal);
			}
			else {
				res.second = ins_name_conflict;
				return res;
			}
		}

		//if we are here then a really new link was inserted
		//if(res.second == ins_ok) {
		//insert into inodep index
		bs_leaf& leaf = const_cast< bs_leaf& >(*ip_idx_.insert(bs_leaf(l, is_persistent)));
		node_key::key_state ks = node_key::key_bad;

		//try to generate key and insert into custom_index
		if(sort_ && !dynamic_cast< const restrict_types* >(sort_.get())) {
			ks = leaf.update_key(sort_);
			if(ks == node_key::key_ok) {
				//block on custom index guard
				bs_mutex::scoped_lock lk(ci_guard_);
				cust_idx_.insert(leaf);
				//listen to changes of object pointed to by given link
				if(!lt_) lt_ = new leaf_tracker(*this);
				l->subscribe(bs_link::data_changed, lt_);
			}
			else
				res.second = ins_bad_cust_idx;
		}
		//}

		// assign parent
		leaf.link_->set_parent(self_);

		//fire signal that leaf was added
		if(self_ && emit_signal)
			self_->fire_signal(bs_node::leaf_added, l);
		return res;
	}

	//------------------------------ low-level erase operations --------------------------------------------------------
	inline void erase_from_idx(const sp_link& l, name_idx_tag) {
		leafs_.erase(l);
	}

	template< class index_tag >
	inline void erase_from_idx(const sp_link& l, index_tag tag) {
		typename index_tag::type::iterator pos = peek_by_name(l, tag);
		if(pos != index(tag).end())
			index(tag).erase(pos);
	}

	void erase_leaf(const name_index_t::iterator& pos, name_idx_tag tag) {
		sp_link dying = iter2link(pos, tag);
		erase_from_idx(dying, cust_idx_tag());
		erase_from_idx(dying, ip_idx_tag());
		leafs_.erase(pos);
	}

	void erase_leaf(const cust_index_t::iterator& pos, cust_idx_tag tag) {
		sp_link dying = iter2link(pos, tag);
		erase_from_idx(dying, name_idx_tag());
		{	//exclusive erase for custom index
			bs_mutex::scoped_lock lk(ci_guard_);
			cust_idx_.erase(pos);
		}
		erase_from_idx(dying, ip_idx_tag());
	}

	void erase_leaf(const inodep_index_t::iterator& pos, ip_idx_tag tag) {
		sp_link dying = iter2link(pos, tag);
		erase_from_idx(dying, name_idx_tag());
		erase_from_idx(dying, cust_idx_tag());
		ip_idx_.erase(pos);
	}

	template< class index_tag >
	bool rem_leaf(const typename index_tag::type::iterator& pos, index_tag tag, bool emit_signal = true) {
		//idx_traits< index_t > tag;
		if(pos == index(tag).end()) return false;
		if(iter2leaf(pos, tag)->is_persistent()) return false;

		//save link
		sp_link dying = iter2link(pos, tag);
		//if(dying->is_persistent()) return false;
		//stop tracking leaf's object
		dying->unsubscribe(bs_link::data_changed, lt_);
		// forget parent
		dying->set_parent(NULL);

		//physically delete leaf
		erase_leaf(pos, tag);

		//fire signal that leaf was deleted
		if(self_ && emit_signal)
			self_->fire_signal(bs_node::leaf_deleted, dying);
		return true;
	}

	//------------------------------------- remove link ----------------------------------------------------------------
	ulong rem_link(const sp_link& l, name_idx_tag tag, bool = true) {
		name_index_t::iterator p_victim = find(l, tag);
		if(rem_leaf(p_victim, tag))
			return 1;
		else return 0;
	}

	template< class index_t >
	ulong erase_range(index_t& idx, const pair< typename index_t::iterator, typename index_t::iterator >& rng) {
		typedef idx_traits< index_t > tag;
		ulong cnt = 0;
		if(rng.first != idx.end()) {
			for(typename index_t::iterator pos = rng.first; pos != rng.second; ++pos) {
				if(rem_leaf(pos, tag()))
					++cnt;
			}
		}
		return cnt;
	}

	template< class index_tag >
	ulong rem_link(const sp_link& l, index_tag tag, bool match_name) {
		pair< typename index_tag::type::iterator, typename index_tag::type::iterator > rng;
		if(match_name) {
			rng.first = find(l, tag, true);
			rng.second = rng.first;
			if(rng.first != index(tag).end())
				++rng.second;
		}
		else
			rng = equal_range(l, tag);

		return erase_range(index(tag), rng);
	}

	//remove all links pointing to the same object
	ulong rem_link(const sp_obj& obj) {
		return rem_link(new bs_link(obj, ""), ip_idx_tag(), false);
	}

	bool rem_link(const string& name) {
		return rem_link(bs_link::dumb_link(name), name_idx_tag(), true) > 0;
	}

	ulong rem_link(const sort_traits::key_ptr& k) {
		pair< cust_index_t::iterator, cust_index_t::iterator > diers = cust_idx_.equal_range(bs_leaf(node_key(k)));
		return erase_range(cust_idx_, diers);
	}

	//--------------------------------------- rename operation ---------------------------------------------------------
	bool rename(const sp_link& l, const std::string& new_name) {
		string old_name = l->name();
		//make dumb link with new name
		sp_link dumb_new = bs_link::dumb_link(new_name);
		//check for name conflict
		if(leafs_.find(dumb_new) == leafs_.end()) {
			// remove old link from name index
			inodep_index_t::iterator pleaf = peek_by_name(l, ip_idx_tag());
			leafs_.erase(l);
			// insert link with new name
			l->rename(new_name);
			leafs_.insert(l);
			// update custom index
			update_custom_idx(const_cast< ip_idx_tag::val_t& >(*pleaf));

			//fire signal that leaf was renamed
			if(self_)
				self_->fire_signal(bs_node::leaf_renamed, l);
			return true;
		}
		else return false;
	}

	//---------------------------- clear tree implementation -----------------------------------------------------------
	ulong clear() {
		return erase_range(leafs_, make_pair(leafs_.begin(), leafs_.end()));
	}

	//----------------------------- pause and continue listening to leafs ----------------------------------------------
	void stop_leafs_tracking() const {
		for(inodep_index_t::const_iterator pos = ip_idx_.begin(), end = ip_idx_.end(); pos != end; ++pos)
			iter2link(pos, ip_idx_tag())->unsubscribe(bs_link::data_changed, lt_);
	}

	void start_leafs_tracking() {
		//first we need to rebuild custom index
		set_sort(sort_);
		//subscribe to unlock signals
		for(inodep_index_t::const_iterator pos = ip_idx_.begin(), end = ip_idx_.end(); pos != end; ++pos)
			iter2link(pos, ip_idx_tag())->subscribe(bs_link::data_changed, lt_);
	}

	//------------------------------------------ sorting operations ----------------------------------------------------
	s_traits_ptr get_sort() const {
		return sort_;
	}

	bool set_sort(const s_traits_ptr& new_sort) {
		sort_ = new_sort;
		if(!sort_) {
			stop_leafs_tracking();
			//clear custom index
			bs_mutex::scoped_lock lk(ci_guard_);
			cust_idx_.clear();
			return true;
		}
//		else if(sort_->sort_name() == new_sort->sort_name())
//			return false;

		//try to build custom index
		if(!dynamic_cast< const restrict_types* >(sort_.get())) {
			//obtain exclusive custom index lock
			bs_mutex::scoped_lock lk(ci_guard_);
			cust_index_t new_cidx;
			try {
				for(inodep_index_t::iterator p = ip_idx_.begin(), end = ip_idx_.end(); p != end; ++p)
					new_cidx.insert(bs_leaf(p->link_, new_sort, p->is_persistent()));
			}
			catch(const bs_exception&) {
				return false;
			}
			//custom index build successful
			cust_idx_ = new_cidx;
		}

		return true;
	}

	//update custom index method
	bool update_custom_idx(bs_leaf& l) {
		if(cust_idx_.empty()) return false;
		//custom index lock should be already obtained in object listener!

		//cout << "changing cust_idx for leaf '" << l.link_->name() << "'" << endl;
		//search for given leaf
		cust_index_t::iterator pl = peek_by_name< cust_index_t >(l, cust_idx_tag());
		if(pl != cust_idx_.end()) {
			//we've found a leaf - rebuild key
			cust_idx_.erase(pl);

			if(l.update_key(sort_) == node_key::key_ok) {
				cust_idx_.insert(l);
				//DEBUG!
				//cout << save_l->link_->data().get() << ": link '" << save_l->link_->name() << "' cust_idx updated" << endl;
				return true;
			}
		}
		return false;
	}

	// true if persistence was really set
	template< class key_t >
	bool set_persistence(const key_t& key, bool persistent) const {
		inodep_index_t::iterator pl = find(key, ip_idx_tag());
		if(pl != ip_idx_.end()) {
			// we can remove constness and modify bs_leaf inplace, because we don't touch sorting key
			const_cast< bs_leaf& >(*pl).set_persistence(persistent);
			return true;
		}
		return false;
	}

	template< class key_t >
	bool is_persistent(const key_t& key) const {
		inodep_index_t::iterator pl = find(key, ip_idx_tag());
		if(pl != ip_idx_.end())
			return pl->is_persistent();
		else return false;
	}
};

//----------- static variables instantiation ----------------------------------------------
boost::detail::atomic_count bs_node::node_impl::unique_cnt_s(0);
//const bs_link bs_node::node_impl::bs_leaf::nil_link_s = *bs_link::dumb_link("__bs_nil_link_tag__");

//============================ n_iterator implementation ===========================================
class bs_node::n_iterator::ni_impl {
public:
	struct iter_backend {
		virtual const sp_link& link() const = 0;
		virtual leaf_ptr leaf() const = 0;
		virtual void operator++() = 0;
		virtual void operator--() = 0;
		virtual bool operator==(const iter_backend&) = 0;
		virtual void assign(const iter_backend&) = 0;
		virtual int index_sn() const = 0;
		virtual bool is_end() const = 0;
		//access to node_impl
		virtual const node_impl* nimpl() const = 0;

		//iter_backend() {}

		//iter_backend(const iter_backend& ib) {
		//	assign(ib);
		//}

		virtual ~iter_backend() {};
	};

	enum initial_pos {
		pos_unspecified = 0,
		pos_begin,
		pos_end
	};

	template< class index_t >
	struct iter_backend_impl : public iter_backend {
//		typedef backend_traits< index_t > traits;
//		typedef typename traits::iter_t iter_t;

		typedef typename index_t::const_iterator iter_t;
		typedef idx_traits< index_t > tag_t;

		node_impl const* nimpl_;
		iter_t pos_;

		//put pos_ to the beginning of it's container
		iter_backend_impl(const node_impl* nimpl = NULL, initial_pos where = pos_unspecified)
			: nimpl_(nimpl)
		{
			if(nimpl_) {
				if(where == pos_begin)
					pos_ = nimpl_->index(tag_t()).begin();
					//pos_ = traits::begin(nimpl_);
				else if(where == pos_end)
					pos_ = nimpl_->index(tag_t()).end();
					//pos_ = traits::end(nimpl_);
			}
		}

		//full construction
		iter_backend_impl(const iter_t& pos, const node_impl* nimpl) : nimpl_(nimpl), pos_(pos)  {}

		//construct from iter_backend using assign
		iter_backend_impl(const iter_backend& ib) : nimpl_(ib.nimpl()) {
			//assign iterator
			assign(ib);
		}

		const node_impl* nimpl() const { return nimpl_; }

		const sp_link& link() const {
			return iter2link(pos_, tag_t());
			//return traits::link(pos_);
		}

		leaf_ptr leaf() const {
			if(nimpl_)
				return nimpl_->iter2leaf(pos_, tag_t());
			else return leaf_ptr::nil_el;
		}

		void operator++() {
			++pos_;
		}
		void operator--() {
			--pos_;
		}

		bool operator==(const iter_backend& ib) {
			if(ib.index_sn() == index_sn())
				return (pos_ == static_cast< const iter_backend_impl& >(ib).pos_);
			else if(!is_end() && !ib.is_end())
				return leaf() == ib.leaf();
			//else if(is_end())
			//	return true;
			else return false;
		}

		int index_sn() const {
			return (int)tag_t::tag;
		}

		void assign(const iter_backend& ib) {
			//check if ib isn't NULL
			if(!ib.nimpl()) {
				//assert(ib.nimpl());
				return;
			}
			nimpl_ = ib.nimpl();

			//check if lhs is of the same type
			//const iter_backend_impl* p_bi = dynamic_cast< const iter_backend_impl* >(&ib);
			//if(p_bi)
			if(ib.index_sn() == index_sn())
				pos_ = static_cast< const iter_backend_impl& >(ib).pos_;
			else if(!ib.is_end())
				pos_ = nimpl_->leaf2iter(ib.leaf(), tag_t());
				//pos_ = nimpl_->find(ib.link(), tag_t());
			else
				pos_ = nimpl_->index(tag_t()).end();

		}

		bool is_end() const {
			return pos_ == nimpl_->index(tag_t()).end();
		}
	};


	//iter_backend* pos_;
	st_smart_ptr< iter_backend > pos_;

	//default ctor
	//template< class index_t >
	ni_impl(index_type idx_t = name_idx) {
		//pos_ = new iter_backend_impl< idx_traits< index_t >::tag >;
		switch(idx_t) {
			default:
			case bs_node::name_idx:
				pos_ = new iter_backend_impl< name_index_t >;
				break;
			case bs_node::custom_idx:
				pos_ = new iter_backend_impl< cust_index_t >;
				break;
			case bs_node::inodep_idx:
				pos_ = new iter_backend_impl< inodep_index_t >;
				break;
		}
	}

	//construct using node
	//template< class index_t >
	ni_impl(const bs_node& d, index_type idx_t = name_idx, initial_pos where = pos_begin) {
		//pos_ = new iter_backend_impl< idx_traits< index_t >::tag >(d.pimpl_);
		switch(idx_t) {
			default:
			case bs_node::name_idx:
				pos_ = new iter_backend_impl< name_index_t >(d.pimpl_, where);
				break;
			case bs_node::custom_idx:
				pos_ = new iter_backend_impl< cust_index_t >(d.pimpl_, where);
				break;
			case bs_node::inodep_idx:
				pos_ = new iter_backend_impl< inodep_index_t >(d.pimpl_, where);
				break;
		}
	}

	//templated ctor using given iterator
	template< class iter_t, class index_tag >
	ni_impl(const node_impl* nimpl, const iter_t& src, index_tag)
		: pos_(new iter_backend_impl< typename index_tag::type >(src, nimpl))
	{
		//DEBUG
		//index_type t = (index_type)iter2idx_tag< iter_t >::tag;
		//cout << t << endl;
		//pos_ = new iter_backend_impl< conversion< typename backend_traits< true >::iter_t, iter_t >::same_type >(src, nimpl);
	}

	//ni_impl(iter_backend* ib) : pos_(ib) {}

	//copy ctor
	ni_impl(const ni_impl& i) {
		assert(i.pos_);
		switch(i.pos_->index_sn()) {
			case name_idx:
				pos_ = new iter_backend_impl< name_index_t >(*i.pos_);
				break;
			case custom_idx:
				pos_ = new iter_backend_impl< cust_index_t >(*i.pos_);
				break;
			case inodep_idx:
				pos_ = new iter_backend_impl< inodep_index_t >(*i.pos_);
				break;
		}

//		if(i.pos_->index_sn() == 0)
//			pos_ = new iter_backend_impl< false >(*i.pos_);
//		else
//			pos_ = new iter_backend_impl< true >(*i.pos_);
	}

	sp_link link() const {
		return pos_->link();
	}

	leaf_ptr leaf() const {
		return pos_->leaf();
	}

	index_type index_id() const {
		return (index_type)pos_->index_sn();
	}

private:
//	template< class iter_t >
//	struct iter2idx_tag {
//		enum { tag = (index_type)(
//			conversion< typename backend_traits< name_index_t >::iter_t, iter_t >::same_type * name_idx +
//			conversion< typename backend_traits< cust_index_t >::iter_t, iter_t >::same_type * custom_idx +
//			conversion< typename backend_traits< inodep_index_t >::iter_t, iter_t >::same_type * inodep_idx
//			)};
//	};
};	//end of ni_impl class

template< class index_t >
bs_node::n_iterator bs_node::node_impl::si2ni(const typename index_t::const_iterator& i) const {
	return n_iterator(new n_iterator::ni_impl(this, i, idx_traits< index_t >()));
}

template< class index_t, class iter_t >
bs_node::n_range bs_node::node_impl::sr2nr(const pair< iter_t, iter_t >& rng) const {
	return bs_node::n_range(
		new n_iterator::ni_impl(this, typename index_t::const_iterator(rng.first), idx_traits< index_t >()),
		new n_iterator::ni_impl(this, typename index_t::const_iterator(rng.second), idx_traits< index_t >())
	);
}

//default ctor
bs_node::n_iterator::n_iterator(index_type idx_t)
: pimpl_(new ni_impl(idx_t))
{}

//bs_node::n_iterator::n_iterator(const bs_node& d, index_type idx_t)
//: pimpl_(new ni_impl(d, idx_t))
//{}

bs_node::n_iterator::n_iterator(ni_impl* pimpl)
: pimpl_(pimpl)
{}

bs_node::n_iterator::n_iterator(const n_iterator& src)
: pimpl_(new ni_impl(*src.pimpl_))
{}

bs_node::n_iterator::~n_iterator() {
	//if(pimpl_) delete pimpl_;
}

bs_node::n_iterator::reference bs_node::n_iterator::operator*() const {
	return *(pimpl_->link());
}

bs_node::n_iterator::pointer bs_node::n_iterator::operator->() const {
	return pimpl_->link();
}

bs_node::n_iterator& bs_node::n_iterator::operator++() {
	++(*pimpl_->pos_);
	return *this;
}

bs_node::n_iterator bs_node::n_iterator::operator++(int) {
	n_iterator tmp(*this);
	++(*pimpl_->pos_);
	return tmp;
}

bs_node::n_iterator& bs_node::n_iterator::operator--() {
	--(*pimpl_->pos_);
	return *this;
}

bs_node::n_iterator bs_node::n_iterator::operator--(int) {
	n_iterator tmp(*this);
	--(*pimpl_->pos_);
	return tmp;
}

bool bs_node::n_iterator::operator==(const n_iterator& i) const {
	return *pimpl_->pos_ == *i.pimpl_->pos_;
}

bool bs_node::n_iterator::operator!=(const n_iterator& i) const {
	return !(*this == i);
}

sp_link bs_node::n_iterator::get() const {
	return pimpl_->link();
}

sp_obj bs_node::n_iterator::data() const {
	return pimpl_->link()->data();
}

sp_inode bs_node::n_iterator::inode() const {
	return pimpl_->link()->inode();
}

void bs_node::n_iterator::swap(n_iterator& i)
{
	std::swap(pimpl_, i.pimpl_);
}

bs_node::n_iterator& bs_node::n_iterator::operator=( const n_iterator& i )
{
	//assignemnt through swap
	n_iterator(i).swap(*this);
	return *this;
}

bs_node::index_type bs_node::n_iterator::index_id() const {
	return pimpl_->index_id();
}

bool bs_node::n_iterator::is_persistent() const {
	return pimpl_->leaf()->is_persistent();
}

void bs_node::n_iterator::set_persistence(bool persistent) const {
	pimpl_->leaf()->set_persistence(persistent);
}

sp_node bs_node::n_iterator::container() const {
	const bs_node::node_impl* p_nimpl = pimpl_->pos_->nimpl();
	if(p_nimpl)
		return p_nimpl->self_;
	else return NULL;
}

//===================================== bs_node  implementation ========================================================

//destructor
bs_node::~bs_node()
{}

//copy ctor implementation
bs_node::bs_node(const bs_node& src)
	: bs_refcounter(src), objbase(src), pimpl_(new node_impl(*src.pimpl_), mutex(), bs_static_cast())
{
	pimpl_.lock()->self_ = this;
}

//ctors
bs_node::bs_node(bs_type_ctor_param param /* = NULL */)
	: bs_refcounter(), objbase(BS_SIGNAL_RANGE(bs_node)), pimpl_((node_impl*)NULL, mutex(), bs_static_cast())
{
//	ostringstream def_name;
//	def_name << "node" << node_impl::next_unique_val();

	smart_ptr< str_data_table > sp_vt(param, bs_dynamic_cast());
	if(!sp_vt) {
		//bs_name_ = def_name.str();
		pimpl_ = new node_impl;
	}
	else {
		s_traits_ptr s;

		//bs_name_ = sp_vt->extract_value< std::string >("name", def_name.str());
		s = sp_vt->extract_value< s_traits_ptr >("sort", NULL);
		pimpl_ = new node_impl(s);
	}
	pimpl_.lock()->self_ = this;
}

bs_node::bs_node(const s_traits_ptr& srt)
	: objbase(BS_SIGNAL_RANGE(bs_node)), pimpl_(new node_impl(srt), mutex(), bs_static_cast())
{
	pimpl_.lock()->self_ = this;
}

blue_sky::sp_node bs_node::create_node(const s_traits_ptr& srt)
{
	lsmart_ptr< smart_ptr< str_data_table > > npar(BS_KERNEL.create_object(str_data_table::bs_type(), true));
	//smart_ptr< str_data_table > sp_dt = give_kernel::Instance().create_object(str_data_table::bs_type());
	//lsmart_ptr< smart_ptr< str_data_table > > npar(sp_dt);
	//lsmart_ptr< kernel::str_dt_ptr > npar(give_kernel::Instance().pert_str_dt(bs_type()));
	assert(npar);
	npar->ss< s_traits_ptr >("sort") = srt;

	return BS_KERNEL.create_object(bs_node::bs_type(), false, npar.get());
}

bs_node::n_iterator bs_node::begin(index_type idx_t) const {
	//use_name_idx |= (pimpl_->cust_idx_.size() == 0);
	return n_iterator(new n_iterator::ni_impl(*this, idx_t));
}

bs_node::n_iterator bs_node::end(index_type idx_t) const {
	return n_iterator(new n_iterator::ni_impl(*this, idx_t, n_iterator::ni_impl::pos_end));

//	use_name_idx |= (pimpl_->cust_idx_.size() == 0);
//	if(use_name_idx)
//		return n_iterator(new n_iterator::ni_impl( pimpl_, pimpl_->leafs_.end() ));
//	else
//		return n_iterator(new n_iterator::ni_impl( pimpl_, pimpl_->cust_idx_.end() ));
}

ulong bs_node::size() const {
	return static_cast< ulong >(pimpl_->leafs_.size());
}

bool bs_node::empty() const {
	return static_cast< ulong >(pimpl_->leafs_.empty());
}

bs_node::s_traits_ptr bs_node::get_sort() const {
	return pimpl_->get_sort();
}

bool bs_node::set_sort(const s_traits_ptr& s) const {
	return pimpl_.lock()->set_sort(s);
}

void bs_node::clear() const {
	//pimpl_->clear();
}

ulong bs_node::count(const sort_traits::key_ptr& k) const {
	return pimpl_->count(k);
}

ulong bs_node::count(const sp_obj& obj) const {
	return pimpl_->count(obj);
}

bs_node::n_iterator bs_node::find(const std::string& name, index_type idx_t) const {
	switch(idx_t) {
		default:
		case name_idx:
			return pimpl_->si2ni< name_index_t >(pimpl_->find(name, name_idx_tag()));
		case custom_idx:
			return pimpl_->si2ni< cust_index_t >(pimpl_->find(name, cust_idx_tag()));
		case inodep_idx:
			return pimpl_->si2ni< inodep_index_t >(pimpl_->find(name, ip_idx_tag()));
	}
}

bs_node::n_iterator bs_node::find(const sp_link& l, bool match_name, index_type idx_t) const {
	switch(idx_t) {
		default:
		case name_idx:
			return pimpl_->si2ni< name_index_t >(pimpl_->find(l, name_idx_tag(), match_name));
		case custom_idx:
			return pimpl_->si2ni< cust_index_t >(pimpl_->find(l, cust_idx_tag(), match_name));
		case inodep_idx:
			return pimpl_->si2ni< inodep_index_t >(pimpl_->find(l, ip_idx_tag(), match_name));
	}
}

bs_node::n_range bs_node::equal_range(const sort_traits::key_ptr& k) const {
	return pimpl_->sr2nr< cust_index_t >(pimpl_->equal_range(k));
}

bs_node::n_range bs_node::equal_range(const sp_link& l, index_type idx_t) const {
	switch(idx_t) {
		default:
		case name_idx:
			return pimpl_->sr2nr< name_index_t >(pimpl_->equal_range(l, name_idx_tag()));
		case custom_idx:
			return pimpl_->sr2nr< cust_index_t >(pimpl_->equal_range(l, cust_idx_tag()));
		case inodep_idx:
			return pimpl_->sr2nr< inodep_index_t >(pimpl_->equal_range(l, ip_idx_tag()));
	}
}

bs_node::insert_ret_t bs_node::insert(const sp_obj& obj, const std::string& name, bool is_persistent) const {
	insert_ret_t res = pimpl_.lock()->add_link(bs_link::create(obj, name), is_persistent, false);
	return res;
}

bs_node::insert_ret_t bs_node::insert(const sp_link& l, bool is_persistent) const {
	insert_ret_t res = pimpl_.lock()->add_link(l, is_persistent, false);
	return res;
}

void bs_node::insert(const n_iterator first, const n_iterator last) const {
	for(n_iterator pos = first; pos != last; ++pos)
		insert(pos.get());
}

ulong bs_node::erase(const sort_traits::key_ptr& k) const {
	return pimpl_.lock()->rem_link(k);
}

ulong bs_node::erase(const sp_link& l, bool match_name, index_type idx_t) const {
	switch(idx_t) {
		default:
		case name_idx:
			return pimpl_.lock()->rem_link(l, name_idx_tag(), match_name);
		case custom_idx:
			return pimpl_.lock()->rem_link(l, cust_idx_tag(), match_name);
		case inodep_idx:
			return pimpl_.lock()->rem_link(l, ip_idx_tag(), match_name);
	}
}

ulong bs_node::erase(const sp_obj& obj) const {
	return pimpl_.lock()->rem_link(obj);
}

ulong bs_node::erase(const std::string& name) const {
	return pimpl_.lock()->rem_link(name);
}

ulong bs_node::erase(n_iterator pos) const {
	//always search by name
	return pimpl_.lock()->rem_link(pos->name());

//	switch(pos.index_id()) {
//		default:
//		case name_idx:
//			return pimpl_.lock()->rem_link(pos.get(), name_idx_tag(), true);
//		case custom_idx:
//			return pimpl_.lock()->rem_link(pos.get(), cust_idx_tag(), true);
//		case inodep_idx:
//			return pimpl_.lock()->rem_link(pos.get(), ip_idx_tag(), true);
//	}
}

ulong bs_node::erase(n_iterator first, n_iterator second) const {
	ulong cnt = 0;
	for(n_iterator pos(first); pos != second; ++pos)
		cnt += erase(pos);
	return cnt;
}

bool bs_node::rename(const std::string& old_name, const std::string& new_name) const {
	return pimpl_.lock()->rename(bs_link::dumb_link(old_name), new_name);
}

bool bs_node::rename(const n_iterator& pos, const std::string& new_name) const {
	return pimpl_.lock()->rename(pos.get(), new_name);
}

void bs_node::start_leafs_tracking() const {
	pimpl_.lock()->start_leafs_tracking();
}

void bs_node::stop_leafs_tracking() const {
	pimpl_->stop_leafs_tracking();
}

bool bs_node::set_persistence(const std::string& link_name, bool persistent) const {
	return pimpl_->set_persistence(link_name, persistent);
}

bool bs_node::set_persistence(const sp_link& l, bool persistent) const {
	return pimpl_->set_persistence(l, persistent);
}

bool bs_node::is_persistent(const std::string& link_name) const {
	return pimpl_->is_persistent(link_name);
}

bool bs_node::is_persistent(const sp_link& l) const {
	return pimpl_->is_persistent(l);
}

//------------------------- sort_traits implementation -----------------------------------------------------------------
bool bs_node::sort_traits::accepts(const sp_link& l) const {
	return (l.get() != NULL);
}

bs_node::sort_traits::types_v bs_node::sort_traits::accept_types() const {
	types_v ret;
	ret.push_back(objbase::bs_type());
	return ret;
}

bool bs_node::restrict_types::no_key::sort_order(const key_ptr&) const { return false; }

bs_node::restrict_types::key_ptr bs_node::restrict_types::key_generator(const sp_link&) const { return NULL; }

//------------------------- bs_node BlueSky macro ----------------------------------------------------------------------

BLUE_SKY_TYPE_STD_COPY(bs_node);
BLUE_SKY_TYPE_STD_CREATE(bs_node);
BLUE_SKY_TYPE_IMPL_SHORT(bs_node, objbase, "The node of BlueSky data storage");

}	//end of namespace blue_sky

