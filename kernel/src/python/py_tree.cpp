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
#include "py_bs_exports.h"
#include "py_smart_ptr.h"
#include "py_pair_converter.h"

#include <boost/python/overloads.hpp>
#include <boost/python/iterator.hpp>
#include <boost/python/enum.hpp>
#include <iterator>

namespace blue_sky { namespace python {

using namespace boost::python;

namespace {
using namespace std;

// sort_traits wrapper
class sort_traits_pyw :
	public bs_node::sort_traits,
	public wrapper< bs_node::sort_traits >
{
public:
	const char* sort_name() const {
		return this->get_override("sort_name")();
	}

	key_ptr key_generator(const sp_link& l) const {
		return this->get_override("key_generator")();
	}

	bool accepts(const sp_link& l) const {
		if(override f = this->get_override("accepts"))
			return f(l);
		else
			return bs_node::sort_traits::accepts(l);
	}

	bool def_accepts(const sp_link& l) const {
		return bs_node::sort_traits::accepts(l);
	}

	types_v accept_types() const {
		if(override f = this->get_override("accept_types"))
			return f();
		else
			return bs_node::sort_traits::accept_types();
	}

	types_v def_accept_types() const {
		return bs_node::sort_traits::accept_types();
	}
};

// sort_traits wrapper
class restrict_types_pyw :
	public bs_node::restrict_types,
	public wrapper< bs_node::restrict_types >
{
public:
	const char* sort_name() const {
		return this->get_override("sort_name")();
	}

	key_ptr key_generator(const sp_link& l) const {
		if(override f = this->get_override("key_generator"))
			return f(l);
		else
			return bs_node::restrict_types::key_generator(l);
	}

	key_ptr def_key_generator(const sp_link& l) const {
		return bs_node::restrict_types::key_generator(l);
	}

	bool accepts(const sp_link& l) const {
		return this->get_override("accepts")();
	}

	types_v accept_types() const {
		return this->get_override("accept_types")();
	}
};

// sort_traits::key_type wrapper
struct key_type_pyw : public bs_node::sort_traits::key_type,
	public wrapper< bs_node::sort_traits::key_type >
{
	bool sort_order(const key_ptr& k) const {
		return this->get_override("sort_order");
	}
};

// reflect Python object for n_iterator::__iter__
bs_node::n_iterator& reflect_pyobj(bs_node::n_iterator& self) {
	return self;
}

sp_link n_iter_next(bs_node::n_iterator& i) {
	// extract n_iterator
	//bs_node::n_iterator& i = extract< bs_node::n_iterator& >(self);
	if(!i.container() || i == i.container()->end()) {
		PyErr_SetNone(PyExc_StopIteration);
		throw_error_already_set();
	}
	else {
		return (i++).get();
	}
	return NULL;
}

// make create_node w/o arguments overload
sp_node create_node() {
	return bs_node::create_node();
}

// we need offset function in order to properly build Qt tree
bs_node::n_iterator::difference_type offset(const bs_node& n, bs_node::n_iterator i) {
	// extract node from object
	//sp_node n = extract< sp_node >(self);
	if(i.container() != &n) return -1;
	return distance(n.begin(), i);
}

template< class key_t >
bs_node::n_iterator::difference_type offset(const bs_node& n, key_t obj_key) {
	// extract node from object
	//sp_node n = extract< sp_node >(self);
	//if(!n) return -1;
	bs_node::n_iterator p_obj = n.find(obj_key);
	if(p_obj != n.end())
		return distance(n.begin(), p_obj);
	else
		return -1;
}

bs_node::n_iterator ss(const bs_node& n, ulong idx) {
	// extract node from object
	//sp_node n = extract< sp_node >(self);
	if(idx >= n.size()) return n.end();
	else {
		bs_node::n_iterator res = n.begin();
		advance(res, idx);
		return res;
	}
}

template< class key_t >
sp_link getitem(const bs_node& n, key_t key) {
	bs_node::n_iterator p_obj = n.find(key);
	if(p_obj == n.end()) {
		PyErr_SetString(PyExc_IndexError, "No such element");
		throw_error_already_set();
	}
	return p_obj.get();
}

template< >
sp_link getitem< ulong >(const bs_node& n, ulong idx) {
	if(idx >= n.size()) {
		PyErr_SetString(PyExc_IndexError, "Index out of range");
		throw_error_already_set();
	}
	return ss(n, idx).get();
}

template< class key_t >
ulong delitem(const bs_node& n, key_t key) {
	return n.erase(key);
}

} 	// eof hidden namespace

// bs_node def argument substitution
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(find1_overl, find, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(find2_overl, find, 1, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(equal_range_overl, equal_range, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(insert1_overl, insert, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(insert2_overl, insert, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(erase_overl, erase, 1, 3);
//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(create_overl, create_node, 0, 1);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(begin_overl, begin, 0, 1);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(end_overl, end, 0, 1);

// exporting function
void py_bind_tree() {
	namespace bp = boost::python;
	// overloads
	// count
	ulong (bs_node::*count1)(const bs_node::sort_traits::key_ptr&) const = &bs_node::count;
	ulong (bs_node::*count2)(const sp_obj&) const = &bs_node::count;
	// find
	bs_node::n_iterator (bs_node::*find1)(const std::string&, bs_node::index_type) const = &bs_node::find;
	bs_node::n_iterator (bs_node::*find2)(const sp_link&, bool, bs_node::index_type) const = &bs_node::find;
	// equal_range
	bs_node::n_range (bs_node::*eq_range1)(const bs_node::sort_traits::key_ptr&) const = &bs_node::equal_range;
	bs_node::n_range (bs_node::*eq_range2)(const sp_link&, bs_node::index_type) const = &bs_node::equal_range;
	// insert
	bs_node::insert_ret_t (bs_node::*insert1)(const sp_obj&, const std::string&, bool is_persistent) const =
		&bs_node::insert;
	bs_node::insert_ret_t (bs_node::*insert2)(const sp_link&, bool) const = &bs_node::insert;
	void (bs_node::*insert3)(bs_node::n_iterator, bs_node::n_iterator) const = &bs_node::insert;
	// erase
	ulong (bs_node::*erase1)(bs_node::n_iterator) const = &bs_node::erase;
	ulong (bs_node::*erase2)(bs_node::n_iterator, bs_node::n_iterator) const = &bs_node::erase;
	ulong (bs_node::*erase3)(const bs_node::sort_traits::key_ptr&) const = &bs_node::erase;
	ulong (bs_node::*erase4)(const sp_link&, bool, bs_node::index_type) const = &bs_node::erase;
	ulong (bs_node::*erase5)(const sp_obj&) const = &bs_node::erase;
	ulong (bs_node::*erase6)(const std::string&) const = &bs_node::erase;
	// rename
	bool (bs_node::*rename1)(const std::string&, const std::string&) const = &bs_node::rename;
	bool (bs_node::*rename2)(const bs_node::n_iterator&, const std::string&) const = &bs_node::rename;
	// is_persistent
	bool (bs_node::*is_persistent1)(const sp_link&) const = &bs_node::is_persistent;
	bool (bs_node::*is_persistent2)(const std::string&) const = &bs_node::is_persistent;
	// set__persistence
	bool (bs_node::*set_persistence1)(const sp_link&, bool) const = &bs_node::set_persistence;
	bool (bs_node::*set_persistence2)(const std::string&, bool) const = &bs_node::set_persistence;
	// offset
	bs_node::n_iterator::difference_type (*offset1)(const bs_node&, bs_node::n_iterator) = &offset;
	bs_node::n_iterator::difference_type (*offset2)(const bs_node&, const std::string&) = &offset;
	bs_node::n_iterator::difference_type (*offset3)(const bs_node&, sp_link) = &offset;

	// bs_node binding
	scope node_scope = class_<
		bs_node,
		smart_ptr< bs_node >,
		bases< objbase >,
		boost::noncopyable
		>
	("node", no_init)
		.def("__iter__", &bs_node::begin, begin_overl())
		.def("__len__", &bs_node::size)
		.def("__getitem__", &getitem< ulong >)
		.def("__getitem__", &getitem< const std::string& >)
		.def("__getitem__", &getitem< sp_link >)
		.def("__delitem__", &delitem< sp_link >)
		.def("__delitem__", &delitem< sp_obj >)
		.def("__delitem__", &delitem< const std::string& >)
		.def("begin", &bs_node::begin, begin_overl())
		.def("end", &bs_node::end, end_overl())
		.def("create", &bs_node::create_node)
		.def("create", &create_node)
		.staticmethod("create")
		.def("bs_type", &bs_node::bs_type)
		.staticmethod("bs_type")
		.def("bs_resolve_type", &bs_node::bs_resolve_type)
		.def("size", &bs_node::size)
		.def("empty", &bs_node::empty)
		.def("clear", &bs_node::clear)
		.def("count", count1)
		.def("count", count2)
		.def("find", find1, find1_overl())
		.def("find", find2, find2_overl())
		.def("equal_range", eq_range1)
		.def("equal_range", eq_range2, equal_range_overl())
		.add_property("sort", &bs_node::get_sort, &bs_node::set_sort)
		.def("insert", insert1, insert1_overl())
		.def("insert", insert2, insert2_overl())
		.def("insert", insert3)
		.def("erase", erase1)
		.def("erase", erase2)
		.def("erase", erase3)
		.def("erase", erase4, erase_overl())
		.def("erase", erase5)
		.def("erase", erase6)
		.def("rename", rename1)
		.def("rename", rename2)
		.def("start_leafs_tracking", &bs_node::start_leafs_tracking)
		.def("stop_leafs_tracking", &bs_node::stop_leafs_tracking)
		.def("is_node", &bs_node::is_node)
		.staticmethod("is_node")
		.def("is_persistent", is_persistent1)
		.def("is_persistent", is_persistent2)
		.def("set_persistence", set_persistence1)
		.def("set_persistence", set_persistence2)
		.def("offset", offset1)
		.def("offset", offset2)
		.def("offset", offset3)
		.def("ss", &ss)
	;

	// register conversion to sp_obj
	implicitly_convertible< sp_node, sp_obj >();

	// signals enum
	enum_< bs_node::signal_codes >("signals")
		.value("leaf_added", bs_node::leaf_added)
		.value("leaf_deleted", bs_node::leaf_deleted)
		.value("leaf_moved", bs_node::leaf_moved)
		.value("leaf_renamed", bs_node::leaf_renamed)
		.export_values()
	;

	// export sort_traits
	{
		//typedef sort_traits_pyw< bs_node::sort_traits > sort_traits_pyw_t;
		scope srt_traits_scope = class_<
			sort_traits_pyw,
			boost::noncopyable
			>
		("sort_traits")
			.def("sort_name", pure_virtual(&bs_node::sort_traits::sort_name))
			.def("key_generator", pure_virtual(&bs_node::sort_traits::key_generator))
			.def("accepts", &bs_node::sort_traits::accepts, &sort_traits_pyw::def_accepts)
			.def("accept_types", &bs_node::sort_traits::accept_types, &sort_traits_pyw::def_accept_types)
		;

		class_<
			key_type_pyw,
			st_smart_ptr< key_type_pyw >,
			boost::noncopyable
			>
		("key_type")
			.def("sort_order", pure_virtual(&bs_node::sort_traits::key_type::sort_order))
		;
		// register smart_ptr conversions
		implicitly_convertible< st_smart_ptr< key_type_pyw >, bs_node::sort_traits::key_type::key_ptr >();
		register_ptr_to_python<  bs_node::sort_traits::key_type::key_ptr >();
	}

	//typedef sort_traits_pyw< bs_node::restrict_types > restrict_types_pyw_t;
	class_<
		restrict_types_pyw,
		bases< bs_node::sort_traits >,
		boost::noncopyable
		>
	("restrict_types")
		.def("key_generator", &bs_node::restrict_types::key_generator, &restrict_types_pyw::def_key_generator)
		.def("accepts", pure_virtual(&bs_node::restrict_types::accepts))
		.def("accept_types", pure_virtual(&bs_node::restrict_types::accept_types))
	;

	// n_iterator binding
	class_<
		bs_node::n_iterator
		>
	("n_iterator", init< optional< bs_node::index_type > >())
		.def("__iter__", &reflect_pyobj, return_internal_reference< >())
		.def("next", &n_iter_next)
		.def("swap", &bs_node::n_iterator::swap)
		.def_readonly("index_id", &bs_node::n_iterator::index_id)
		.def("get", &bs_node::n_iterator::get)
		.def_readonly("link", &bs_node::n_iterator::get)
		.def_readonly("inode", &bs_node::n_iterator::inode)
		.def_readonly("data", &bs_node::n_iterator::data)
		.add_property("persistent", &bs_node::n_iterator::is_persistent,
				&bs_node::n_iterator::set_persistence)
		// operators
		.def(self == self)
		.def(self != self)
	;

	// insert_ret_t binding
	typedef bspy_converter< pair_traits< bs_node::insert_ret_t > > insert_ret_converter;
	insert_ret_converter::register_to_py();
}

}}	// eof namespace blue_sky::python

