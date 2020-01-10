/// @file
/// @author uentity
/// @date 22.04.2019
/// @brief BS node Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/common.h>
#include <bs/tree/node.h>

#include <boost/uuid/uuid_io.hpp>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

NAMESPACE_BEGIN(blue_sky::python)
using namespace tree;
using Key = node::Key;
using InsertPolicy = node::InsertPolicy;

/*-----------------------------------------------------------------------------
 *  hidden details
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()

// helpers to omit code duplication
// ------- contains
template<Key K>
bool contains(const node& N, const std::string& key) {
	return N.find(key, K) != N.end<>();
}
bool contains_link(const node& N, const sp_link& l) {
	return N.find(l->id()) != N.end<>();
}
bool contains_obj(const node& N, const sp_obj& obj) {
	return N.find(obj->id(), Key::OID) != N.end<>();
}

// ------- find
template<Key K, bool Throw = true>
auto find(const node& N, const std::string& key) -> sp_link {
	auto r = N.find(key, K);
	if(r != N.end<>()) return *r;
	if(Throw) {
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
	return nullptr;
}

template<bool Throw = true>
sp_link find_obj(const node& N, const sp_obj& obj) {
	return find<Key::OID, Throw>(N, obj->id());
}

auto find_by_idx(const node& N, const long idx) {
	// support for Pythonish indexing from both ends
	if(auto positive_idx = idx < 0 ? N.size() + idx : std::size_t(idx); positive_idx < N.size())
		return std::next(N.begin(), positive_idx);
	throw py::key_error("Index out of bounds");
}

// ------- index
template<Key K>
auto index(const node& N, const std::string& key) -> std::size_t {
	return N.index(key, K);
}
auto index_link(const node& N, const sp_link& l) {
	return N.index(l->id());
}
auto index_obj(const node& N, const sp_obj& obj) {
	return N.index(obj->id(), Key::OID);
}

// ------- deep search
template<Key K>
auto deep_search(const node& N, const std::string& key) -> sp_link {
	return N.deep_search(key, K);
}
auto deep_search_obj(const node& N, const sp_obj& obj) {
	return N.deep_search(obj->id(), Key::OID);
}

// ------- erase
template<Key K>
void erase(node& N, const std::string& key) {
	N.erase(key, K);
}
void erase_link(node& N, const sp_link& l) {
	N.erase(l->id());
}
void erase_obj(node& N, const sp_obj& obj) {
	N.erase(obj->id(), Key::OID);
}
void erase_idx(node& N, const long idx) {
	// support for Pythonish indexing from both ends
	if(std::size_t(std::abs(idx)) > N.size())
		throw py::key_error("Index out of bounds");
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
		.value("DenyDupOID", InsertPolicy::DenyDupOID)
		.value("Merge", InsertPolicy::Merge)
		//.export_values();
	;
	//py::implicitly_convertible<int, InsertPolicy>();
	//py::implicitly_convertible<long, InsertPolicy>();

	node_pyface
		BSPY_EXPORT_DEF(node)
		.def(py::init<>())
		.def("__len__", &node::size)
		.def("__iter__",
			[](const node& N) { return py::make_iterator(N.begin(), N.end()); },
			py::keep_alive<0, 1>()
		)

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
			return *find_by_idx(N, idx);
		}, "link_idx"_a)
		// search by link ID
		.def("__getitem__", &find<Key::ID>, "lid"_a)
		.def("find",        &find<Key::ID, false>, "lid"_a, "Find link with given ID")
		// search by object instance
		.def("__getitem__", &find_obj<>, "obj"_a)
		.def("find",        &find_obj<false>, "obj"_a, "Find link to given object")
		// search by object ID
		.def("find_oid",    &find<Key::OID, false>, "oid"_a, "Find link to object with given ID")
		// search by link name
		.def("find_name",   &find<Key::Name, false>, "link_name"_a, "Find link with given name")

		// obtain index in custom order from link ID
		.def("index", &index<Key::ID>, "lid"_a, "Find index of link with given ID")
		// ...from link itself
		.def("index", &index_link, "link"_a, "Find index of given link")
		// ... from object
		.def("index", &index_obj, "obj"_a, "Find index of link to given object")
		// ... from OID
		.def("index_oid",  &index<Key::OID>, "oid"_a, "Find index of link to object with given ID")
		// ... from link name
		.def("index_name", &index<Key::Name>, "name"_a, "Find index of link with given name")

		// deep search by object
		.def("deep_search", &deep_search_obj, "obj"_a, "Deep search for link to given object")
		// deep search by link ID
		.def("deep_search", &deep_search<Key::ID>, "lid"_a, "Deep search for link with given ID")
		// deep search by link name
		.def("deep_search_name", &deep_search<Key::Name>, "link_name"_a, "Deep search for link with given name")
		// deep search by object ID
		.def("deep_search_oid", &deep_search<Key::OID>, "oid"_a, "Deep search for link to object with given ID")

		.def("equal_range", [](const node& N, const std::string& link_name) {
			auto r = N.equal_range(link_name);
			return py::make_iterator(r.first, r.second);
		}, py::keep_alive<0, 1>(), "link_name"_a)
		.def("equal_range_oid", [](const node& N, const std::string& oid) {
			auto r = N.equal_range_oid(oid);
			return py::make_iterator(r.first, r.second);
		}, py::keep_alive<0, 1>(), "OID"_a)
		.def("equal_type", [](const node& N, const std::string& type_id) {
			auto r = N.equal_type(type_id);
			return py::make_iterator(r.first, r.second);
		}, py::keep_alive<0, 1>(), "obj_type_id"_a)

		// insert given link
		.def("insert", [](node& N, const sp_link& l, InsertPolicy pol = InsertPolicy::AllowDupNames) {
			return N.insert(l, uint(pol)).second;
		}, "link"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert given link")
		// insert link at given index
		.def("insert", [](node& N, const sp_link& l, const long idx, InsertPolicy pol = InsertPolicy::AllowDupNames) {
			return N.insert(l, idx, pol).second;
		}, "link"_a, "idx"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert link at given index")
		// insert hard link to given object
		.def("insert", [](node& N, std::string name, sp_obj obj, InsertPolicy pol = InsertPolicy::AllowDupNames) {
			return N.insert(std::move(name), std::move(obj), pol).second;
		}, "name"_a, "obj"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert hard link to given object")

		// erase by given index
		.def("__delitem__", &erase_idx, "idx"_a)
		.def("erase",       &erase_idx, "idx"_a, "Erase link with given index")
		// erase given link
		.def("__delitem__", &erase_link, "link"_a)
		.def("erase",       &erase_link, "link"_a, "Erase given link")
		// erase by given link ID
		.def("__delitem__", &erase<Key::ID>, "lid"_a)
		.def("erase",       &erase<Key::ID>, "lid"_a, "Erase link with given ID")
		// erase by object instance
		.def("__delitem__", &erase_obj, "obj"_a)
		.def("erase",       &erase_obj, "obj"_a, "Erase links to given object")
		// erase by link name
		.def("erase_name", &erase<Key::Name>, "link_name"_a, "Erase links with given name")
		// erase by OID
		.def("erase_oid",  &erase<Key::OID>, "oid"_a, "Erase links to object with given ID")
		// erase by obj type
		.def("erase_type", &erase<Key::Type>, "obj_type_id"_a, "Erase all links pointing to objects of given type")

		// misc container-related functions
		.def_property_readonly("size", &node::size)
		.def_property_readonly("empty", &node::empty)
		.def("clear", &node::clear, "Clears all node contents")
		.def("keys", [](const node& N, Key ktype = Key::ID) {
			if(ktype == Key::ID) {
				// convert UUIDs to string representation
				auto keys = N.keys<>();
				std::vector<std::string> res;
				res.reserve(keys.size());
				for(const auto& k : keys)
					res.emplace_back(boost::uuids::to_string(k));
				return res;
			}
			else {
				switch(ktype) {
				default:
				case Key::Name :
					return N.keys<Key::Name>();
				case Key::OID :
					return N.keys<Key::OID>();
				case Key::Type :
					return N.keys<Key::Type>();
				};
			}
		}, "key_type"_a = Key::ID)

		// link rename
		// by ID or OID or name
		.def("rename", py::overload_cast<const std::string&, std::string, Key, bool>(&node::rename),
			"key"_a, "new_name"_a, "key_meaning"_a = Key::ID, "all"_a = false,
			"Rename link with given key (ID, OID or link name)")
		// by link instance (extracts ID)
		.def("rename", [](node& N, const sp_link& l, std::string new_name) {
			return N.rename(l->id(), std::move(new_name));
		}, "link"_a, "new_name"_a, "Rename given link")
		// by index offset
		.def("rename", [](node& N, const long idx, std::string new_name) {
			return N.rename(find_by_idx(N, idx), std::move(new_name));
		}, "idx"_a, "new_name"_a, "Rename link with given index")
		// by object instance
		.def("rename", [](node& N, const sp_obj& obj, std::string new_name, bool all = false) {
			return N.rename(obj->id(), std::move(new_name), Key::OID, all);
		}, "obj"_a, "new_name"_a, "all"_a = false, "Rename link(s) to given object")

		// misc API
		.def("accepts", &node::accepts, "Check if node accepts given link")
		.def("accept_object_types", &node::accept_object_types,
			"Set white list of object types that node will accept"
		)
		.def_property_readonly("allowed_object_types", &node::allowed_object_types,
			"Returns white list of object types that node will accept"
		)

		.def_property_readonly("handle", &node::handle,
			"Returns a single link that owns this node in overall tree"
		)
		.def("propagate_owner", &node::propagate_owner, "deep"_a = false,
			"Set owner of all contained links to this node (if deep, fix owner in entire subtree)"
		)

		// events subscrition
		.def("subscribe", &node::subscribe, "event_cb"_a, "events"_a = Event::All)
		.def("unsubscribe", &node::unsubscribe, "event_cb_id"_a)
		.def("disconnect", &node::disconnect, "deep"_a = true, "Stop receiving messages from tree")
	;
}

NAMESPACE_END(blue_sky::python)
