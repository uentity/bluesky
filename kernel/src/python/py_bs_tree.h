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

#ifndef _PY_BS_TREE_H
#define _PY_BS_TREE_H

#include "bs_tree.h"
#include "py_bs_object_base.h"

namespace blue_sky {
namespace python {

class BS_API py_bs_node : public py_objbase {
public:
	py_bs_node(const py_objbase&);
	py_bs_node(const sp_obj&);
	py_bs_node(const sp_node&);
	py_bs_node(const py_bs_node&);

	class BS_API py_n_iterator : public std::iterator<
		std::bidirectional_iterator_tag,
		py_bs_link, ptrdiff_t,
		py_bs_link, py_bs_link >
	{
		friend class py_bs_node;

	public:
		py_n_iterator(bs_node::index_type idx_t);
		py_n_iterator(const bs_node::n_iterator&);
		py_n_iterator(const py_n_iterator&);
		~py_n_iterator();

		reference operator*() const;

		pointer operator->() const;

		py_n_iterator& operator++();
		py_n_iterator operator++(int);

		py_n_iterator& operator--();
		py_n_iterator operator--(int);

		bool operator ==(const py_n_iterator&) const;
		bool operator !=(const py_n_iterator&) const;

		void swap(py_n_iterator& i);
		py_n_iterator& operator=(const py_n_iterator& i);

		bs_node::index_type index_id() const;
		py_bs_link get() const;
		py_bs_inode inode() const;
		py_objbase data() const;

	private:
		mutable bs_node::n_iterator niter;
		//mutable py_bs_link pylink;
	};

	typedef py_n_iterator iterator;

	typedef std::reverse_iterator< py_n_iterator > py_rn_iterator;
	typedef std::pair< py_n_iterator, py_n_iterator > py_n_range;
	typedef std::pair< py_n_iterator, int > py_insert_ret_t;

	//py_n_iterator begin() const;
	//py_n_iterator end() const;
	py_n_iterator begin(bs_node::index_type idx_t = bs_node::name_idx) const;
	py_n_iterator end(bs_node::index_type idx_t = bs_node::name_idx) const;

	py_rn_iterator rbegin(bs_node::index_type idx_t = bs_node::name_idx) const;
	py_rn_iterator rend(bs_node::index_type idx_t = bs_node::name_idx) const;

	ulong size() const;
	bool empty() const;

	void clear() const;
	//ulong count(const sort_traits::key_ptr& k) const;
	ulong count(const py_objbase& obj) const;

	py_n_iterator find1(const std::string& name, size_t) const;//bs_node::index_type idx_t = bs_node::name_idx) const;
	py_n_iterator find2(const py_bs_link& l, bool match_name = true, size_t idx_t = (size_t)bs_node::name_idx) const;

	//bs_node::n_range equal_range(const bs_node::sort_traits::key_ptr& k) const;
	py_n_range equal_range(const py_bs_link& l, bs_node::index_type idx_t = bs_node::name_idx) const;

	//s_traits_ptr get_sort() const;
	//bool set_sort(const s_traits_ptr& s) const;

	py_insert_ret_t insert1(const py_objbase& obj, const std::string& name, bool force = false) const;
	py_insert_ret_t insert2(const py_bs_link& l, bool force = false) const;
	void insert3(const py_n_iterator first, const py_n_iterator last) const;

	//ulong erase(const sort_traits::key_ptr& k) const;
	ulong erase1(const py_bs_link& l, bool match_name = true, bs_node::index_type idx_t = bs_node::name_idx) const;
	ulong erase2(const py_objbase& obj) const;
	ulong erase3(const std::string& name) const;
	ulong erase4(py_n_iterator pos) const;
	ulong erase5(py_n_iterator first, py_n_iterator second) const;

	bool rename1(const std::string& old_name, const std::string& new_name) const;
	bool rename2(const py_n_iterator& pos, const std::string& new_name) const;

	static py_bs_node create_node(/*const s_traits_ptr& srt = NULL*/);

	static bool is_node(const py_objbase &obj);

private:
	sp_node spnode;
};

//typedef py_bs_node::py_n_iterator (py_bs_node::*find1)(const std::string&, bs_node::index_type idx_t) const;
//typedef py_bs_node::py_n_iterator (py_bs_node::*find2)(const py_bs_link&, bool match_name, bs_node::index_type idx_t) const;

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _PY_BS_TREE_H
