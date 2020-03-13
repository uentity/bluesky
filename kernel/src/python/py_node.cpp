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

#include <boost/uuid/uuid_io.hpp>

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
template<Key K, bool Throw = true>
auto find(const node& N, std::string key) -> link {
	if(auto r = N.find(std::move(key), K))
		return r;

	if constexpr(Throw) {
		py::gil_scoped_acquire();

		std::string msg = "Node doesn't contain link ";
		switch(K) {
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

template<bool Throw = true>
link find_obj(const node& N, const sp_obj& obj) {
	return find<Key::OID, Throw>(N, obj->id());
}

auto find_by_idx(const node& N, long idx) {
	// support for Pythonish indexing from both ends
	if(auto positive_idx = idx < 0 ? N.size() + idx : std::size_t(idx); positive_idx < N.size())
		return N.find(positive_idx);

	py::gil_scoped_acquire();
	throw py::key_error("Index out of bounds");
}

// ------- index
template<Key K>
auto index(const node& N, std::string key) -> std::size_t {
	if(auto i = N.index(std::move(key), K)) return *i;
	py::gil_scoped_acquire();
	throw py::index_error();
}
auto index_link(const node& N, const link& l) {
	if(auto i = N.index(l.id())) return *i;
	py::gil_scoped_acquire();
	throw py::index_error();
}
auto index_obj(const node& N, const sp_obj& obj) {
	if(auto i = N.index(obj->id(), Key::OID)) return *i;
	py::gil_scoped_acquire();
	throw py::index_error();
}

// ------- deep search
template<Key K>
auto deep_search(const node& N, std::string key) -> link {
	return N.deep_search(std::move(key), K);
}

auto deep_search_obj(const node& N, const sp_obj& obj) {
	return N.deep_search(obj->id(), Key::OID);
}

// ------- equal_range
template<Key K>
auto equal_range(const node& N, std::string key) -> links_v {
	return N.equal_range(std::move(key), K);
	//py::gil_scoped_acquire();
	//return make_container_iterator(std::move(res));
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
	py::class_<node, objbase, std::shared_ptr<node>> node_pyface(m, "node");

	// export node's Key enum
	py::enum_<Key>(node_pyface, "Key")
		.value("ID", Key::ID)
		.value("OID", Key::OID)
		.value("Name", Key::Name)
		.value("Type", Key::Type)
		.value("AnyOrder", Key::AnyOrder)
	;
	// export node's insert policy
	py::enum_<InsertPolicy>(node_pyface, "InsertPolicy", py::arithmetic())
		.value("AllowDupNames", InsertPolicy::AllowDupNames)
		.value("DenyDupNames", InsertPolicy::DenyDupNames)
		.value("RenameDup", InsertPolicy::RenameDup)
		.value("Merge", InsertPolicy::Merge)
	;

	node_pyface
		BSPY_EXPORT_DEF(node)
		.def(py::init<>())
		.def("__len__", &node::size, nogil)
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
		.def("__getitem__", [](const node& N, const long idx) {
			return find_by_idx(N, idx);
		}, "link_idx"_a, nogil)
		// search by link ID
		.def("__getitem__", &find<Key::ID>, "lid"_a, nogil)
		.def("find",        &find<Key::ID, false>, "lid"_a, "Find link with given ID", nogil)
		// search by object instance
		.def("__getitem__", &find_obj<>, "obj"_a, nogil)
		.def("find",        &find_obj<false>, "obj"_a, "Find link to given object", nogil)
		// search by object ID
		.def("find_oid",    &find<Key::OID, false>, "oid"_a, "Find link to object with given ID", nogil)
		// search by link name
		.def("find_name",   &find<Key::Name, false>, "link_name"_a, "Find link with given name", nogil)

		// obtain index in custom order from link ID
		.def("index", &index<Key::ID>, "lid"_a, "Find index of link with given ID", nogil)
		// ...from link itself
		.def("index", &index_link, "link"_a, "Find index of given link", nogil)
		// ... from object
		.def("index", &index_obj, "obj"_a, "Find index of link to given object", nogil)
		// ... from OID
		.def("index_oid",  &index<Key::OID>, "oid"_a, "Find index of link to object with given ID", nogil)
		// ... from link name
		.def("index_name", &index<Key::Name>, "name"_a, "Find index of link with given name", nogil)

		// deep search by object
		.def("deep_search", &deep_search_obj, "obj"_a, "Deep search for link to given object", nogil)
		// deep search by link ID
		.def("deep_search", &deep_search<Key::ID>, "lid"_a, "Deep search for link with given ID", nogil)
		// deep search by link name
		.def("deep_search_name", &deep_search<Key::Name>, "link_name"_a,
			"Deep search for link with given name", nogil)
		// deep search by object ID
		.def("deep_search_oid", &deep_search<Key::OID>, "oid"_a,
			"Deep search for link to object with given ID", nogil)
		// deep serach by specified key type
		.def("deep_search", py::overload_cast<std::string, Key>(&node::deep_search, py::const_),
			"key"_a, "key_meaning"_a, "Deep search by key with specified treatment", nogil)
		// deep equal range
		.def("deep_equal_range", &node::deep_equal_range,
			"key"_a, "key_meaning"_a, "Deep search all links with given key & treatment", nogil)

		.def("equal_range", &equal_range<Key::Name>, "link_name"_a, nogil)
		.def("equal_range_oid", &equal_range<Key::OID>, "OID"_a, nogil)
		.def("equal_type", &equal_range<Key::Type>, "obj_type_id"_a, nogil)

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
		// erase by link name
		.def("erase_name", &erase<Key::Name>, "link_name"_a, "Erase links with given name", nogil)
		// erase by OID
		.def("erase_oid",  &erase<Key::OID>, "oid"_a, "Erase links to object with given ID", nogil)
		// erase by obj type
		.def("erase_type", &erase<Key::Type>, "obj_type_id"_a, "Erase all links pointing to objects of given type", nogil)

		// misc container-related functions
		.def_property_readonly("size", &node::size, nogil)
		.def_property_readonly("empty", &node::empty, nogil)
		.def("clear", &node::clear, "Clears all node contents", nogil)
		.def("keys", &node::skeys, "key_type"_a = Key::ID, "ordering"_a = Key::AnyOrder, nogil)

		// link rename
		.def("rename", py::overload_cast<std::string, std::string>(&node::rename),
			"old_name"_a, "new_name"_a, "Rename all links with given old_name", nogil)
		// by index offset
		.def("rename", py::overload_cast<std::size_t, std::string>(&node::rename),
			"idx"_a, "new_name"_a, "Rename link with given index", nogil)
		// by ID
		.def("rename", py::overload_cast<lid_type, std::string>(&node::rename),
			"lid"_a, "new_name"_a, "Rename link with given ID", nogil)

		// misc API
		.def_property_readonly("handle", &node::handle,
			"Returns a single link that owns this node in overall tree"
		)
		.def("propagate_owner", &node::propagate_owner, "deep"_a = false,
			"Set owner of all contained links to this node (if deep, fix owner in entire subtree)"
		)

		// events subscrition
		.def("subscribe", &node::subscribe, "event_cb"_a, "events"_a = Event::All, nogil)
		.def("unsubscribe", &node::unsubscribe, "event_cb_id"_a, nogil)
		.def("disconnect", &node::disconnect, "deep"_a = true, "Stop receiving messages from tree", nogil)
	;
}

NAMESPACE_END(blue_sky::python)
