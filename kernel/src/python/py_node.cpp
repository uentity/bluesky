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
template<Key K>
bool contains(const node& N, std::string key) {
	return N.index(std::move(key), K).has_value();
}
bool contains_link(const node& N, const link& l) {
	return N.index(l.id()).has_value();
}
bool contains_obj(const node& N, const sp_obj& obj) {
	return N.index(obj->id(), Key::OID).has_value();
}

// ------- find
template<bool Throw = true>
auto find(const node& N, std::string key, Key meaning) -> link {
	if(auto r = N.find(std::move(key), meaning))
		return r;

	if constexpr(Throw) {
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
	return {};
}

template<Key K, bool Throw = true>
auto find(const node& N, std::string key) -> link {
	return find<Throw>(N, std::move(key), K);
}

template<bool Throw = true>
link find_obj(const node& N, const sp_obj& obj) {
	return find<Throw>(N, obj->id(), Key::OID);
}

auto find_by_idx(const node& N, long idx) {
	// support for Pythonish indexing from both ends
	if(auto positive_idx = idx < 0 ? N.size() + idx : std::size_t(idx); positive_idx < N.size())
		return N.find(positive_idx);

	py::gil_scoped_acquire();
	throw py::key_error("Index out of bounds");
}

// ------- index
// [NOTE] don't throw by default
template<Key K, bool Throw = false>
auto index(const node& N, std::string key) -> py::int_ {
	if(auto i = N.index(std::move(key), K))
		return *i;
	if constexpr(Throw) {
		py::gil_scoped_acquire();
		throw py::index_error();
	}
	else return py::none();
}

auto index_obj(const node& N, const sp_obj& obj) {
	return index<Key::OID, false>(N, obj->id());
}

auto index_link(const node& N, const link& l) {
	return N.index(l.id());
}

// ------- deep search
template<Key K>
auto deep_search(const node& N, std::string key) -> link {
	return N.deep_search(std::move(key), K);
}

auto deep_search_obj(const node& N, const sp_obj& obj) {
	return N.deep_search(obj->id(), Key::OID);
}

// ------- erase
template<Key K>
void erase(node& N, const std::string& key) {
	N.erase(key, K);
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
		.def("__len__", &node::size, nogil)
		.def_property_readonly("is_nil", &node::is_nil, nogil)
		.def_property_readonly_static("nil", &node::nil)

		// [FIXME] segfaults if `gil_scoped_release` is applied as call guard
		.def("__iter__", [](const node& N) {
			py::gil_scoped_release();
			auto res = N.leafs();
			return make_container_iterator(std::move(res));
		})

		.def("leafs", &node::leafs, "Key"_a = Key::AnyOrder, "Return snapshot of node content", nogil)

		// check by link ID
		.def("__contains__", &contains_link, "link"_a)
		.def("contains",     &contains_link, "link"_a, "Check if node contains given link")
		.def("__contains__", &contains<Key::ID>, "lid"_a)
		.def("contains",     &contains<Key::ID>, "lid"_a, "Check if node contains link with given ID")
		// ... by object
		.def("__contains__", &contains_obj, "obj"_a)
		.def("contains",     &contains_obj, "obj"_a, "Check if node contains link to given object")
		// check by link name
		.def("contains_name", &contains<Key::Name>,
			"link_name"_a, "Check if node contains link with given name")
		// check by object ID
		.def("contains_oid",  &contains<Key::OID>,
			"oid"_a, "Check if node contains link to object with given ID")
		// check by object type
		.def("contains_type", &contains<Key::Type>,
			"obj_type_id"_a, "Check if node contain links to objects of given type")

		// get item by int index
		.def("__getitem__", &find_by_idx, "link_idx"_a, nogil)
		// search by link ID
		.def("__getitem__", &find<Key::ID>, "lid"_a, nogil)
		.def("find",        &find<Key::ID, false>, "lid"_a, "Find link with given ID", nogil)
		// search by object instance
		.def("__getitem__", &find_obj<>, "obj"_a, nogil)
		.def("find",        &find_obj<false>, "obj"_a, "Find link to given object", nogil)
		// search by string key & key treatment
		.def("find", py::overload_cast<std::string, Key>(&node::find, py::const_),
			"key"_a, "key_meaning"_a, "Find link by key with specified treatment", nogil)

		// obtain index in custom order from link ID
		.def("index", &index<Key::ID>, "lid"_a, "Find index of link with given ID", nogil)
		// ...from link itself
		.def("index", &index_link, "link"_a, "Find index of given link", nogil)
		// ... from object
		.def("index", &index_obj, "obj"_a, "Find index of link to given object", nogil)
		// ... from string key with specified treatment
		.def("index", py::overload_cast<std::string, Key>(&node::index, py::const_),
			"key"_a, "key_meaning"_a, "Find index of link with specified key and treatment", nogil)

		// deep search by object
		.def("deep_search", &deep_search_obj, "obj"_a, "Deep search for link to given object", nogil)
		// deep search by link ID
		.def("deep_search", &deep_search<Key::ID>, "lid"_a, "Deep search for link with given ID", nogil)
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
		.def("insert", [](node& N, link l, const long idx, InsertPolicy pol = InsertPolicy::AllowDupNames) {
			return N.insert(std::move(l), idx, pol).second;
		}, "link"_a, "idx"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert link at given index", nogil)
		// insert hard link to given object
		.def("insert", [](node& N, std::string name, sp_obj obj, InsertPolicy pol = InsertPolicy::AllowDupNames) {
			return N.insert(std::move(name), std::move(obj), pol).second;
		}, "name"_a, "obj"_a, "pol"_a = InsertPolicy::AllowDupNames,
		"Insert hard link to given object", nogil)
		.def("insert", [](node& N, links_v ls, InsertPolicy pol) {
			return N.insert(std::move(ls), pol);
		}, "ls"_a, "pol"_a = InsertPolicy::AllowDupNames, "insert bunch of links at once", nogil)

		// erase by given index
		.def("__delitem__", &erase_idx, "idx"_a, nogil)
		.def("erase",       &erase_idx, "idx"_a, "Erase link with given index", nogil)
		// erase given link
		.def("__delitem__", &erase_link, "link"_a, nogil)
		.def("erase",       &erase_link, "link"_a, "Erase given link", nogil)
		// erase by given link ID
		.def("__delitem__", &erase<Key::ID>, "lid"_a, nogil)
		.def("erase",       &erase<Key::ID>, "lid"_a, "Erase link with given ID", nogil)
		// erase by object instance
		.def("__delitem__", &erase_obj, "obj"_a, nogil)
		.def("erase",       &erase_obj, "obj"_a, "Erase links to given object", nogil)
		// generic erase by key & meaning
		.def("erase", py::overload_cast<std::string, Key>(&node::erase, py::const_),
			"key"_a, "key_meaning"_a = Key::Name, "Erase all leafs with given key", nogil)

		// misc container-related functions
		.def_property_readonly("size", &node::size, nogil)
		.def_property_readonly("empty", &node::empty, nogil)
		.def("clear", &node::clear, "Clears all node contents", nogil)

		.def("keys", &node::skeys, "key_type"_a = Key::ID, "ordering"_a = Key::AnyOrder, nogil)
		.def("ikeys", &node::ikeys, "ordering"_a = Key::AnyOrder, nogil)

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

		// misc API
		.def_property_readonly("handle", &node::handle,
			"Returns a single link that owns this node in overall tree"
		)
		//.def("propagate_owner", &node::propagate_owner, "deep"_a = false,
		//	"Set owner of all contained links to this node (if deep, fix owner in entire subtree)"
		//)

		// events subscrition
		.def("subscribe", &node::subscribe, "event_cb"_a, "events"_a = Event::All, nogil)
		.def_static("unsubscribe", &node::unsubscribe, "event_cb_id"_a)
	;

	// node::weak_ptr
	bind_weak_ptr(node_pyface);

	// [TODO] remove it later (added for compatibility)
	node_pyface.attr("Key") = m.attr("Key");
	node_pyface.attr("InsertPolicy") = m.attr("InsertPolicy");
}

NAMESPACE_END(blue_sky::python)
