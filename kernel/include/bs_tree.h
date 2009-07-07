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

#ifndef _BS_TREE
#define _BS_TREE

#include "bs_common.h"
#include "bs_object_base.h"
#include "bs_link.h"

#ifdef _MSC_VER
#pragma warning(push)
//disable complaints about std::iterator have no dll interface
#pragma warning(disable:4275)
#endif

namespace blue_sky {

class BS_API bs_node : public objbase
{
	friend class bs_link;
	friend class kernel;

public:

	typedef smart_ptr< bs_node, true > sp_node;
	//typedef mt_ptr< directory > mp_node;

	enum index_type {
		name_idx = 0,
		custom_idx,
		inodep_idx
	};

	enum insert_result {
		ins_ok = 0,
		ins_name_conflict,
		ins_type_conflict,
		ins_bad_cust_idx,
		ins_unknown_error
	};

	//------------------------------ signals definition ---------------------------
	BLUE_SKY_SIGNALS_DECL_BEGIN(objbase)
		leaf_added,
		leaf_deleted,
		leaf_moved,
		leaf_renamed,
	BLUE_SKY_SIGNALS_DECL_END

	//============================== sort traits ==================================
	struct BS_API sort_traits {
		typedef std::vector< type_descriptor > types_v;
		struct key_type {
			typedef st_smart_ptr< key_type > key_ptr;
			virtual bool sort_order(const key_ptr& k) const = 0;
		};
		typedef key_type::key_ptr key_ptr;

		virtual const char* sort_name() const = 0;
		virtual key_ptr key_generator(const sp_link& l) const = 0;

		virtual bool accepts(const sp_link& l) const;

		virtual types_v accept_types() const;
	};

	struct BS_API restrict_types : public sort_traits {
		struct no_key : public key_type {
			bool sort_order(const key_ptr&) const;
		};

		virtual const char* sort_name() const = 0;
		key_ptr key_generator(const sp_link&) const;

		virtual bool accepts(const sp_link&) const = 0;

		virtual types_v accept_types() const = 0;
	};

	//useful typedefs
	typedef st_smart_ptr< sort_traits > s_traits_ptr;

	//============================== end of sort traits ============================

	//iterator for accessing leafs of current directory
	class BS_API n_iterator : public std::iterator< std::bidirectional_iterator_tag, bs_link, ptrdiff_t,
		sp_link, sp_link::ref_t >
	{
		friend class bs_node;

	private:
		class ni_impl;
		ni_impl* pimpl_;
		n_iterator(ni_impl*);

	public:
		//default ctor
		n_iterator(index_type idx_t = name_idx);
		//after construction n_iterator will point to bs_node::begin()
		//n_iterator(const bs_node&, index_type idx_t = name_idx);
		n_iterator(const n_iterator&);
		~n_iterator();

		reference operator*() const;

		pointer operator->() const;

		n_iterator& operator++();
		n_iterator operator++(int);

		n_iterator& operator--();
		n_iterator operator--(int);

		bool operator ==(const n_iterator&) const;
		bool operator !=(const n_iterator&) const;

		void swap(n_iterator& i);
		n_iterator& operator=(const n_iterator& i);

		index_type index_id() const;
		sp_link get() const;
		sp_inode inode() const;
		sp_obj data() const;
		bool is_persistent() const;
		void set_persistence(bool persistent) const;
	};

	typedef std::reverse_iterator< n_iterator > rn_iterator;
	typedef std::pair< n_iterator, n_iterator > n_range;
	typedef std::pair< n_iterator, int > insert_ret_t;

	//member functions
	//bool is_persistent() const;

	n_iterator begin(index_type idx_t = name_idx) const;
	n_iterator end(index_type idx_t = name_idx) const;

	rn_iterator rbegin(index_type idx_t = name_idx) const {
		return rn_iterator(end(idx_t));
	}
	rn_iterator rend(index_type idx_t = name_idx) const {
		return rn_iterator(begin(idx_t));
	}

	//returns number of leafs in the tree
	ulong size() const;
	//checks if tree is empty
	bool empty() const;

	//deletes all the leafs from the tree
	void clear() const;
	//count leafs that are associated with given key
	ulong count(const sort_traits::key_ptr& k) const;
	//the same, but given the object pointer
	ulong count(const sp_obj& obj) const;

	//returns an iterator addressing the location of leaf with given name
	n_iterator find(const std::string& name, index_type idx_t = name_idx) const;
	//returns an iterator addressing the location of given link
	n_iterator find(const sp_link& l, bool match_name = true, index_type idx_t = name_idx) const;

	//return all leafs corresponding to given key
	n_range equal_range(const sort_traits::key_ptr& k) const;
	//return all leafs corresponding to given link
	n_range equal_range(const sp_link& l, index_type idx_t = name_idx) const;

	//custom sorting operations
	s_traits_ptr get_sort() const;
	bool set_sort(const s_traits_ptr& s) const;

	//leafs addition
	//insert value
	insert_ret_t insert(const sp_obj& obj, const std::string& name, bool is_persistent = false) const;
	insert_ret_t insert(const sp_link& l, bool is_persistent = false) const;
	//insert from iterator range
	void insert(const n_iterator first, const n_iterator last) const;

	//leafs deletion by different keys
	ulong erase(const sort_traits::key_ptr& k) const;
	ulong erase(const sp_link& l, bool match_name = true, index_type idx_t = name_idx) const;
	ulong erase(const sp_obj& obj) const;
	ulong erase(const std::string& name) const;

	//deletiotion of leafs pointed to by iterator
	ulong erase(n_iterator pos) const;
	ulong erase(n_iterator first, n_iterator second) const;

	//renames particular leaf
	bool rename(const std::string& old_name, const std::string& new_name) const;
	bool rename(const n_iterator& pos, const std::string& new_name) const;

	//start monitoring unlock event from leafs objects und update custom index
	void start_leafs_tracking() const;
	//stop leafs tracking
	void stop_leafs_tracking() const;

	//static node creation members
	static sp_node create_node(const s_traits_ptr& srt = NULL);
	//static sp_node create_node(const s_traits_stack& sorts);

	//check if given object is really an instance of bs_node
	static bool is_node(const sp_obj obj) {
		return dynamic_cast< const bs_node* >(obj.get()) != NULL;
	}

	// persistense maintance
	bool is_persistent(const sp_link& link) const;
	bool is_persistent(const std::string& link_name) const;
	// return true if persistence was really altered
	bool set_persistence(const sp_link& link, bool persistent) const;
	bool set_persistence(const std::string& link_name, bool persistent) const;

	//virtual dtor
	virtual ~bs_node();

protected:
	class node_impl;
	smart_ptr< node_impl, false > pimpl_;

	//sort by pointer - useful only in .system folder - really means unsorted
	//static s_traits_ptr sort_by_ptr();

	//ctors
	bs_node(const s_traits_ptr& srt);
	//bs_node(const std::string& name, const s_traits_stack& sorts);

	//leafs persistence maintance
	//void set_persistent(const std::string& leaf_name, bool persistent) const;
	//void set_persistent(const n_iterator& pos, bool persistent) const;

	//void mv(const sp_node& where);

	BLUE_SKY_TYPE_DECL(bs_node)
};

typedef bs_node::sp_node sp_node;

} //namespace blue-sky

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // _BS_TREE
