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
#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

#include "py_bs_tree.h"
#include "py_bs_exports.h"
#include <boost/python/enum.hpp>

namespace blue_sky {
namespace python {

py_bs_node::py_bs_node(const py_objbase &obj)
	: py_objbase(obj),spnode(obj.sp)
{}

py_bs_node::py_bs_node(const sp_obj &obj)
	: py_objbase(obj),spnode(obj)
{}

py_bs_node::py_bs_node(const sp_node &tsp)
	: py_objbase(sp_obj(dynamic_cast<const objbase*>(tsp.get()))),spnode(tsp)
{}

py_bs_node::py_bs_node(const py_bs_node &lnode)
	: py_objbase(sp_obj(dynamic_cast<const objbase*>(lnode.spnode.get()))),spnode(lnode.spnode)
{}

py_bs_node::py_n_iterator::py_n_iterator(bs_node::index_type idx_t)
	: niter(idx_t)//,pylink(NULL)
{}

py_bs_node::py_n_iterator::py_n_iterator(const py_n_iterator &iter)
	: niter(iter.niter)//,pylink(NULL)
{}

py_bs_node::py_n_iterator::py_n_iterator(const bs_node::n_iterator &iter)
	: niter(iter)//,pylink(NULL)
{}

py_bs_node::py_n_iterator::~py_n_iterator() {}

py_bs_node::py_n_iterator::reference py_bs_node::py_n_iterator::operator*() const {
	//pylink = py_bs_link(niter.get());
	//return pylink;
	return py_bs_link(niter.get());
}

py_bs_node::py_n_iterator::pointer py_bs_node::py_n_iterator::operator->() const {
	//pylink = py_bs_link(niter.get());
	//return &pylink;
	return py_bs_link(niter.get());
}

py_bs_node::py_n_iterator &py_bs_node::py_n_iterator::operator++() {
	++niter;
	return *this;
}

py_bs_node::py_n_iterator py_bs_node::py_n_iterator::operator++(int) {
	py_n_iterator tmp(niter);
	++niter;
	return tmp;
}

py_bs_node::py_n_iterator& py_bs_node::py_n_iterator::operator--() {
	--niter;
	return *this;
}

py_bs_node::py_n_iterator py_bs_node::py_n_iterator::operator--(int) {
	py_n_iterator tmp(niter);
	--niter;
	return tmp;
}

bool py_bs_node::py_n_iterator::is_persistent() const {
	return niter.is_persistent();
}

void py_bs_node::py_n_iterator::set_persistence(bool persistent) const {
	return niter.set_persistence(persistent);
}

bool py_bs_node::py_n_iterator::operator ==(const py_bs_node::py_n_iterator &iter) const {
	return iter.niter == niter;
}

bool py_bs_node::py_n_iterator::operator !=(const py_bs_node::py_n_iterator &iter) const {
	return iter.niter != niter;
}

void py_bs_node::py_n_iterator::swap(py_bs_node::py_n_iterator &iter) {
	return niter.swap(iter.niter);
}

py_bs_node::py_n_iterator &py_bs_node::py_n_iterator::operator=(const py_bs_node::py_n_iterator &iter) {
	niter = iter.niter;
	return *this;
}

bs_node::index_type py_bs_node::py_n_iterator::index_id() const {
	return niter.index_id();
}

py_bs_link py_bs_node::py_n_iterator::get() const {
	return py_bs_link(niter.get());
}

py_bs_inode py_bs_node::py_n_iterator::inode() const {
	return py_bs_inode(niter.inode());
}

py_objbase py_bs_node::py_n_iterator::data() const {
	return py_objbase(niter.data());
}

//py_bs_node::py_n_iterator py_bs_node::begin() const {
//	return py_n_iterator(spnode->begin(bs_node::name_idx));
//}

//py_bs_node::py_n_iterator py_bs_node::end() const {
//	return py_n_iterator(spnode->end(bs_node::name_idx));
//}

py_bs_node::py_n_iterator py_bs_node::begin(bs_node::index_type idx_t) const {
	return py_n_iterator(spnode->begin(idx_t));
}

py_bs_node::py_n_iterator py_bs_node::end(bs_node::index_type idx_t) const {
	return py_n_iterator(spnode->end(idx_t));
}

py_bs_node::py_rn_iterator py_bs_node::rbegin(bs_node::index_type idx_t) const {
	return py_rn_iterator(spnode->end(idx_t));
}

py_bs_node::py_rn_iterator py_bs_node::rend(bs_node::index_type idx_t) const {
	return py_rn_iterator(spnode->begin(idx_t));
}

ulong py_bs_node::size() const { return spnode->size(); }
bool py_bs_node::empty() const { return spnode->empty(); }

void py_bs_node::clear() const { spnode->clear(); }
//ulong count(const sort_traits::key_ptr& k) const;
ulong py_bs_node::count(const py_objbase& obj) const {
	return spnode->count(obj.sp);
}

py_bs_node::py_n_iterator py_bs_node::find1(const std::string& name, size_t idx_t) const {//bs_node::index_type idx_t) const {
	return py_n_iterator(spnode->find(name,(bs_node::index_type)idx_t));
}

py_bs_node::py_n_iterator py_bs_node::find2(const py_bs_link& l, bool match_name, size_t idx_t) const {
	return py_n_iterator(spnode->find(l.splink,match_name,(bs_node::index_type)idx_t));
}

//bs_node::n_range equal_range(const bs_node::sort_traits::key_ptr& k) const {
//
//}

py_bs_node::py_n_range py_bs_node::equal_range(const py_bs_link& l, bs_node::index_type idx_t) const {
	bs_node::n_range range(spnode->equal_range(l.splink,idx_t));
	return py_n_range(py_n_iterator(range.first),py_n_iterator(range.second));
}

//s_traits_ptr get_sort() const;
//bool set_sort(const s_traits_ptr& s) const;

py_bs_node::py_insert_ret_t py_bs_node::insert1(const py_objbase& obj, const std::string& name, bool force) const {
	bs_node::insert_ret_t range(spnode->insert(obj.sp,name,force));
	return py_insert_ret_t(py_n_iterator(range.first),range.second);
}

py_bs_node::py_insert_ret_t py_bs_node::insert2(const py_bs_link& l, bool force) const {
	bs_node::insert_ret_t range(spnode->insert(l.splink,force));
	return py_insert_ret_t(py_n_iterator(range.first),range.second);
}

void py_bs_node::insert3(const py_n_iterator first, const py_n_iterator last) const {
	spnode->insert(first.niter,last.niter);
}

//ulong erase(const sort_traits::key_ptr& k) const;
ulong py_bs_node::erase1(const py_bs_link& l, bool match_name, bs_node::index_type idx_t) const {
	return spnode->erase(l.splink,match_name,idx_t);
}

ulong py_bs_node::erase2(const py_objbase& obj) const {
	return spnode->erase(obj.sp);
}

ulong py_bs_node::erase3(const std::string& name) const {
	return spnode->erase(name);
}

ulong py_bs_node::erase4(py_n_iterator pos) const {
	return spnode->erase(pos.niter);
}

ulong py_bs_node::erase5(py_n_iterator first, py_n_iterator second) const {
	return spnode->erase(first.niter,second.niter);
}

bool py_bs_node::rename1(const std::string& old_name, const std::string& new_name) const {
	return spnode->rename(old_name,new_name);
}

bool py_bs_node::rename2(const py_n_iterator& pos, const std::string& new_name) const {
	return spnode->rename(pos.niter,new_name);
}

py_bs_node py_bs_node::create_node(/*const s_traits_ptr& srt = NULL*/) {
	return py_bs_node(bs_node::create_node());
}

bool py_bs_node::is_node(const py_objbase &obj) {
	return bs_node::is_node(obj.sp);
}

bool py_bs_node::is_persistent1(const py_bs_link& link) const {
	return spnode->is_persistent(link.splink);
}

bool py_bs_node::is_persistent2(const std::string& link_name) const {
	return spnode->is_persistent(link_name);
}

// return true if persistence was really altered
bool py_bs_node::set_persistence1(const py_bs_link& link, bool persistent) const {
	return spnode->set_persistence(link.splink, persistent);
}
bool py_bs_node::set_persistence2(const std::string& link_name, bool persistent) const {
	return spnode->set_persistence(link_name, persistent);
}

void py_export_tree() {
	class_<py_bs_node, bases<py_objbase> >("node", init<const py_objbase&>())
		.def("__iter__",boost::python::iterator< py_bs_node >())
		.def("size",&py_bs_node::size)
		.def("empty",&py_bs_node::empty)
		.def("clear",&py_bs_node::clear)
		.def("count",&py_bs_node::count)
		.def("find",&py_bs_node::find1)
		.def("find",&py_bs_node::find2)
		.def("insert",&py_bs_node::insert1)
		.def("insert",&py_bs_node::insert2)
		.def("insert",&py_bs_node::insert3)
		.def("erase",&py_bs_node::erase1)
		.def("erase",&py_bs_node::erase2)
		.def("erase",&py_bs_node::erase3)
		.def("erase",&py_bs_node::erase4)
		.def("erase",&py_bs_node::erase5)
		.def("rename",&py_bs_node::rename1)
		.def("rename",&py_bs_node::rename2)
		.def("is_node",&py_bs_node::is_node)
		.def("is_persistent", &py_bs_node::is_persistent1)
		.def("is_persistent", &py_bs_node::is_persistent2)
		.def("set_persistence", &py_bs_node::set_persistence1)
		.def("set_persistence", &py_bs_node::set_persistence2)
		;

	class_<py_bs_node::py_n_iterator>("py_n_iterator",no_init)
		.def("index_id",&py_bs_node::py_n_iterator::index_id)
		.def("get",&py_bs_node::py_n_iterator::get)
		.def("data",&py_bs_node::py_n_iterator::data)
		.def("is_persistent", &py_bs_node::py_n_iterator::is_persistent)
		.def("set_persistence", &py_bs_node::py_n_iterator::set_persistence)
		;

	class_<std::pair< py_bs_node::py_n_iterator, int > >("insert_ret_t",no_init)
		.add_property("first",&std::pair< py_bs_node::py_n_iterator, int >::first)
		.add_property("second",&std::pair< py_bs_node::py_n_iterator, int >::second);

	def("create_node",&py_bs_node::create_node);

	//enum_<bs_node::signal_codes>("node_signal_codes")
	//	.value("leaf_added",bs_node::leaf_added)
	//	.value("leaf_deleted",bs_node::leaf_deleted)
	//	.value("leaf_moved",bs_node::leaf_moved)
	//	.value("leaf_renamed",bs_node::leaf_renamed)
	//	.export_values();
}

}	//end of namespace blue_sky::python
}	//end of namespace blue_sky
