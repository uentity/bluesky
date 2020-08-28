/// @file
/// @author uentity
/// @date 22.04.2019
/// @brief BS node Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/tree.h>
#include <bs/python/container_iterator.h>

#include "../tree/node_leafs_storage.h"

#include <pybind11/functional.h>
#include <pybind11/chrono.h>

NAMESPACE_BEGIN(blue_sky::python)
using namespace tree;

/*-----------------------------------------------------------------------------
 *  hidden details
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()

// helpers to omit code duplication
// ------- contains
bool contains_lid(const node& N, lid_type key) {
	return N.index(std::move(key)).has_value();
}
bool contains_link(const node& N, const link& l) {
	return N.index(l.id()).has_value();
}
bool contains_obj(const node& N, const sp_obj& obj) {
	return N.index(obj->id(), Key::OID).has_value();
}

bool contains_key(const node& N, std::string key, Key meaning) {
	const auto check_index = [&](Key meaning) {
		return N.index(std::move(key), meaning).has_value();
	};
	switch(meaning) {
	default:
	case Key::ID:
		return check_index(Key::ID);
	case Key::OID:
		return check_index(Key::OID);
	case Key::Name:
		return check_index(Key::Name);
	case Key::Type:
		return check_index(Key::Type);
	}
}

// ------- find
auto throw_find_err(const std::string& key, Key meaning) {
	py::gil_scoped_acquire();

	std::string msg = "Node doesn't contain link ";
	switch(meaning) {
	default:
	case Key::ID:
		msg += "with ID = "; break;
	case Key::OID:
		msg += "to object with ID = "; break;
	case Key::Name:
		msg += "with name = "; break;
	case Key::Type:
		msg += "with type = "; break;
	}
	throw py::key_error(msg + key);
}

template<bool Throw = true>
auto find_lid(const node& N, lid_type key) -> link {
	if(auto r = N.find(std::move(key)))
		return r;

	if constexpr(Throw)
		throw_find_err(to_string(key), Key::ID);
	return {};
}

template<bool Throw = true>
auto find_obj(const node& N, const sp_obj& obj) -> link {
	if(auto r = N.find(obj->id(), Key::OID))
		return r;

	if constexpr(Throw)
		throw_find_err(obj->id(), Key::OID);
	return {};
}

auto find_idx(const node& N, long idx) {
	// support for Pythonish indexing from both ends
	if(auto positive_idx = idx < 0 ? N.size() + idx : std::size_t(idx); positive_idx < N.size())
		return N.find(positive_idx);

	py::gil_scoped_acquire();
	throw py::key_error("Index out of bounds");
}

// ------- index
auto index_obj(const node& N, const sp_obj& obj) -> py::int_ {
	if(auto i = N.index(obj->id(), Key::OID))
		return *i;
	else return py::none();
}
auto index_lid(const node& N, const lid_type& lid) {
	return N.index(lid);
}

auto index_link(const node& N, const link& l) {
	return N.index(l.id());
}

// ------- deep search
auto deep_search_lid(const node& N, const lid_type& key) -> link {
	return N.deep_search(key);
}

auto deep_search_obj(const node& N, const sp_obj& obj) {
	return N.deep_search(obj->id(), Key::OID);
}

// ------- erase
void erase_lid(node& N, const lid_type& key) {
	N.erase(key);
}
void erase_link(node& N, const link& l) {
	N.erase(l.id());
}
void erase_obj(node& N, const sp_obj& obj) {
	N.erase(obj->id(), Key::OID);
}
void erase_idx(node& N, const long idx) {
	// support for Pythonish indexing from both ends
	if(std::size_t(std::abs(idx)) > N.size()) {
		py::gil_scoped_acquire();
		throw py::key_error("Index out of bounds");
	}
	N.erase(idx < 0 ? N.size() + idx : idx);
}

NAMESPACE_END() // hidden ns

void py_bind_node(py::module& m) {
	///////////////////////////////////////////////////////////////////////////////
	//  Node
	//
	auto node_pyface = py::class_<node, engine>(m, "node")
		.def(py::init<>())

		.def("__bool__", [](const node& self) { return (bool)self; }, py::is_operator())
		.def("__len__", py::overload_cast<>(&node::size, py::const_), nogil)
		.def("__iter__", [](const node& N) {
			py::gil_scoped_release();
			auto res = N.leafs();
			return make_container_iterator(std::move(res));
		})

		.def_property_readonly("size_unsafe", [](const node& self) { return self.size(unsafe); })
		.def_property_readonly("empty_unsafe", [](const node& self) { return self.empty(unsafe); })
		.def_property_readonly("handle", &node::handle, "Returns a single link that owns this node")
		.def_property_readonly("is_nil", &node::is_nil)
		.def_property_readonly_static("nil", [](const py::object&) { return node::nil(); })

		.def("leafs", py::overload_cast<Key>(&node::leafs, py::const_),
			"Key"_a = Key::AnyOrder, "Return snapshot of node content", nogil)
		.def("leafs", py::overload_cast<unsafe_t, Key>(&node::leafs, py::const_),
			"Return snapshot of node content", nogil)

		.def("keys", &node::keys, "ordering"_a = Key::AnyOrder, nogil)
		.def("ikeys", &node::ikeys, "ordering"_a = Key::AnyOrder, nogil)
		.def("skeys", &node::skeys, "key_meaing"_a, "ordering"_a = Key::AnyOrder, nogil)

		// check by link ID
		// [NOTE] it's essential to register UUID overload first, because Python native UUID has a
		// convertion operator to `int` (deprecated, but...). Hence, conversion from UUID -> int is
		// tried, warning is printed, ...
		.def("__contains__", &contains_lid, "lid"_a)
		.def("contains",     &contains_lid, "lid"_a, "Check if node contains link with given ID")
		// by link instance
		.def("__contains__", &contains_link, "link"_a)
		.def("contains",     &contains_link, "link"_a, "Check if node contains given link")
		// ... by object
		.def("__contains__", &contains_obj, "obj"_a)
		.def("contains",     &contains_obj, "obj"_a, "Check if node contains link to given object")
		// ... generic version
		.def("contains", &contains_key, "key"_a, "key_meaning"_a, "Check if node contains link with given key")

		// search by link ID
		.def("__getitem__", &find_lid<true>, "lid"_a, nogil)
		.def("find",        &find_lid<false>, "lid"_a, "Find link with given ID", nogil)
		// get item by int index
		.def("__getitem__", &find_idx, "link_idx"_a, nogil)
		// search by object instance
		.def("__getitem__", &find_obj<true>, "obj"_a, nogil)
		.def("find",        &find_obj<false>, "obj"_a, "Find link to given object", nogil)
		// search by string key & key treatment
		.def("find", py::overload_cast<std::string, Key>(&node::find, py::const_),
			"key"_a, "key_meaning"_a, "Find link by key with specified treatment", nogil)

		// obtain index in custom order from link ID
		.def("index", &index_lid, "lid"_a, "Find index of link with given ID", nogil)
		// ...from link itself
		.def("index", &index_link, "link"_a, "Find index of given link", nogil)
		// ... from object
		.def("index", &index_obj, "obj"_a, "Find index of link to given object", nogil)
		// ... from string key with specified treatment
		.def("index", py::overload_cast<std::string, Key>(&node::index, py::const_),
			"key"_a, "key_meaning"_a, "Find index of link with specified key and treatment", nogil)

		// deep search by link ID
		.def("deep_search", &deep_search_lid, "lid"_a, "Deep search for link with given ID", nogil)
		// deep search by object
		.def("deep_search",&deep_search_obj, "obj"_a, "Deep search for link to given object", nogil)
		// deep serach by specified key type
		.def("deep_search", py::overload_cast<std::string, Key>(&node::deep_search, py::const_),
			"key"_a, "key_meaning"_a, "Deep search by key with specified treatment", nogil)

		// equal range
		.def("equal_range", &node::equal_range,
			"key"_a, "key_meaning"_a = Key::Name, "Find all leafs with given key", nogil)

		.def("deep_equal_range", &node::deep_equal_range,
			"key"_a, "key_meaning"_a, "Deep search all links with given key & treatment", nogil)

		// insert given link
		.def("insert", [](node& N, link l, InsertPolicy pol = InsertPolicy::AllowDupNames) {
			return N.insert(std::move(l), pol).second;
		}, "link"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert given link", nogil)
		// insert link at given index
		.def("insert", [](node& N, link l, const long idx, InsertPolicy pol) {
			return N.insert(std::move(l), idx, pol).second;
		}, "link"_a, "idx"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert link at given index", nogil)
		// insert hard link to given object
		.def("insert", [](node& N, std::string name, sp_obj obj, InsertPolicy pol) {
			return N.insert(std::move(name), std::move(obj), pol).second;
		}, "name"_a, "obj"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert hard link to given object", nogil)
		// insert hard link to given node
		.def("insert", [](node& N, std::string name, node n, InsertPolicy pol) {
			return N.insert(std::move(name), std::move(n), pol).second;
		}, "name"_a, "folder"_a, "pol"_a = InsertPolicy::AllowDupNames,
		"Insert hard link to given node (objnode is created internally)", nogil)
		// insert bunch of links
		.def("insert", [](node& N, links_v ls, InsertPolicy pol) {
			return N.insert(std::move(ls), pol);
		}, "ls"_a, "pol"_a = InsertPolicy::AllowDupNames, "insert bunch of links at once", nogil)

		// erase by given link ID
		.def("__delitem__", &erase_lid, "lid"_a, nogil)
		.def("erase",       &erase_lid, "lid"_a, "Erase link with given ID", nogil)
		// erase by given index
		.def("__delitem__", &erase_idx, "idx"_a, nogil)
		.def("erase",       &erase_idx, "idx"_a, "Erase link with given index", nogil)
		// erase given link
		.def("__delitem__", &erase_link, "link"_a, nogil)
		.def("erase",       &erase_link, "link"_a, "Erase given link", nogil)
		// erase by object instance
		.def("__delitem__", &erase_obj, "obj"_a, nogil)
		.def("erase",       &erase_obj, "obj"_a, "Erase links to given object", nogil)
		// generic erase by key & meaning
		.def("erase", py::overload_cast<std::string, Key>(&node::erase, py::const_),
			"key"_a, "key_meaning"_a = Key::Name, "Erase all leafs with given key", nogil)

		// misc container-related functions
		.def("clear", py::overload_cast<>(&node::clear, py::const_), "Clears all node contents", nogil)
		.def("clear", py::overload_cast<launch_async_t>(&node::clear, py::const_),
			"Async clear all node contents")

		// link rename
		.def("rename", py::overload_cast<std::string, std::string>(&node::rename, py::const_),
			"old_name"_a, "new_name"_a, "Rename all links with given old_name", nogil)
		// by index offset
		.def("rename", py::overload_cast<std::size_t, std::string>(&node::rename, py::const_),
			"idx"_a, "new_name"_a, "Rename link with given index", nogil)
		// by ID
		.def("rename", py::overload_cast<lid_type, std::string>(&node::rename, py::const_),
			"lid"_a, "new_name"_a, "Rename link with given ID", nogil)

		.def("rearrange", py::overload_cast<lids_v>(&node::rearrange, py::const_),
			"new_order"_a, "Apply custom order to node")
		.def("rearrange", py::overload_cast<std::vector<std::size_t>>(&node::rearrange, py::const_),
			"new_order"_a, "Apply custom order to node")

		// events subscrition
		.def("subscribe", &node::subscribe, "event_cb"_a, "events"_a = Event::All, nogil)
		.def_static("unsubscribe", &node::unsubscribe, "event_cb_id"_a)
	;

	// node::weak_ptr
	bind_weak_ptr(node_pyface);
}

NAMESPACE_END(blue_sky::python)
