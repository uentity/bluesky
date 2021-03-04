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

#include "kernel_queue.h"
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
template<typename Node>
bool contains_lid(const Node& N, lid_type key) {
	return N.index(std::move(key)).has_value();
}

template<typename Node>
bool contains_link(const Node& N, const link& l) {
	return N.index(l.id()).has_value();
}

bool contains_obj(const node& N, const sp_obj& obj) {
	return obj ? N.index(obj->id(), Key::OID).has_value() : false;
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

template<bool Throw = true, typename Node>
auto find_lid(const Node& N, lid_type key) -> link {
	if(auto r = N.find(std::move(key)))
		return r;

	if constexpr(Throw)
		throw_find_err(to_string(key), Key::ID);
	return {};
}

template<bool Throw = true>
auto find_obj(const node& N, const sp_obj& obj) -> link {
	// sanity
	if(obj) {
		if(auto r = N.find(obj->id(), Key::OID))
			return r;
	}

	if constexpr(Throw) {
		if(obj) throw_find_err(obj->id(), Key::OID);
		else throw py::key_error("Nil object");
	}
	return {};
}

template<typename Node>
auto find_idx(const Node& N, long idx) -> link {
	// support for Pythonish indexing from both ends
	std::size_t positive_idx = [&] {
		if(idx < 0) {
			idx += N.size();
			if(idx < 0) throw py::key_error("Index out of bounds");
		}
		return static_cast<std::size_t>(idx);
	}();

	if(auto res = N.find(positive_idx))
		return res;
	else
		throw py::key_error("Index out of bounds");
}

// ------- index
auto index_obj(const node& N, const sp_obj& obj) -> py::int_ {
	if(obj) {
		if(auto i = N.index(obj->id(), Key::OID))
			return *i;
	}
	return py::none();
}

template<typename Node>
auto index_lid(const Node& N, const lid_type& lid) -> typename Node::existing_index {
	return N.index(lid);
}

template<typename Node>
auto index_link(const Node& N, const link& l) -> typename Node::existing_index {
	return N.index(l.id());
}

// ------- deep search
auto deep_search_lid(const node& N, const lid_type& key) -> link {
	return N.deep_search(key);
}

auto deep_search_obj(const node& N, const sp_obj& obj) {
	return obj ? N.deep_search(obj->id(), Key::OID) : link{};
}

// ------- erase
template<typename Node>
void erase_lid(Node& N, const lid_type& key) {
	N.erase(key);
}

template<typename Node>
void erase_link(Node& N, const link& l) {
	N.erase(l.id());
}

void erase_obj(node& N, const sp_obj& obj) {
	if(obj)
		N.erase(obj->id(), Key::OID);
}

template<typename Node>
void erase_idx(Node& N, const long idx) {
	// support for Pythonish indexing from both ends
	if(std::size_t(std::abs(idx)) > N.size())
		throw py::key_error("Index out of bounds");
	N.erase(idx < 0 ? N.size() + idx : idx);
}

NAMESPACE_END()

void py_bind_node(py::module& m) {
	///////////////////////////////////////////////////////////////////////////////
	// minimal link API present in both bare & normal link
	//
	const auto add_common_api = [](auto& pyn, const auto&... gil) {
		using py_node_type = std::remove_reference_t<decltype(pyn)>;
		using node_type = typename py_node_type::type;

		return pyn
		.def(hash(py::self))
		.def("__bool__", [](const node_type& self) { return (bool)self; }, py::is_operator())

		.def_property_readonly("handle", &node_type::handle, "Returns a single link that owns this node")
		.def_property_readonly("is_nil", &node_type::is_nil)

		.def("__len__", &node_type::size, gil...)
		.def("size", &node_type::size, gil...)
		.def("empty", &node_type::empty, gil...)

		.def("leafs", &node_type::leafs,
			"Key"_a = Key::AnyOrder, "Return snapshot of node content", gil...)

		.def("keys", [](const node& N, Key key_meaning, Key ordering) {
				if(key_meaning == Key::ID)
					return py::cast(N.keys(ordering));
				else if(key_meaning == Key::AnyOrder)
					return py::cast(N.ikeys(ordering));
				else
				   return py::cast(N.skeys(key_meaning, ordering));
			}, "key_meaning"_a = Key::ID, "ordering"_a = Key::AnyOrder,
			"Return keys of `key_meaning` type sorted according to `ordering`"
		)

		// check if node contains key
		// [NOTE] it's essential to register UUID overload first, because Python native UUID has a
		// convertion operator to `int` (deprecated, but...). Hence, conversion from UUID -> int is
		// tried, warning is printed, ...
		.def("__contains__", &contains_lid<node_type>, "lid"_a, gil...)
		.def("contains",     &contains_lid<node_type>, "lid"_a, "Check if node contains link with given ID")
		// by link instance
		.def("__contains__", &contains_link<node_type>, "link"_a, gil...)
		.def("contains",     &contains_link<node_type>, "link"_a, "Check if node contains given link")

		// search by link ID
		.def("__getitem__", &find_lid<true, node_type>, "lid"_a)
		.def("find",        &find_lid<false, node_type>, "lid"_a, "Find link with given ID")
		// get item by int index
		.def("__getitem__", &find_idx<node_type>, "link_idx"_a)

		// obtain index in custom order from link ID
		.def("index", &index_lid<node_type>, "lid"_a, "Find index of link with given ID", gil...)
		// ...from link itself
		.def("index", &index_link<node_type>, "link"_a, "Find index of given link", gil...)

		// insert given link
		.def("insert", [](node_type& N, link l, InsertPolicy pol) {
			return N.insert(std::move(l), pol);
		}, "link"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert given link", gil...)
		// insert bunch of links
		.def("insert", [](node_type& N, links_v ls, InsertPolicy pol) {
			return N.insert(std::move(ls), pol);
		}, "links"_a, "pol"_a = InsertPolicy::AllowDupNames, "insert bunch of links at once", gil...)
		// insert hard link to given object
		.def("insert", [](node_type& N, std::string name, sp_obj obj, InsertPolicy pol) {
			return N.insert(std::move(name), std::move(obj), pol);
		}, "name"_a, "obj"_a, "pol"_a = InsertPolicy::AllowDupNames,
		"Insert hard link to given object", gil...)
		// insert hard link to given node
		.def("insert", [](node_type& N, std::string name, node n, InsertPolicy pol) {
			return N.insert(std::move(name), std::move(n), pol);
		}, "name"_a, "folder"_a, "pol"_a = InsertPolicy::AllowDupNames,
		"Insert hard link to given node (objnode is created internally)", gil...)

		// erase by given link ID
		.def("__delitem__", &erase_lid<node_type>, "lid"_a, gil...)
		.def("erase",       &erase_lid<node_type>, "lid"_a, "Erase link with given ID", gil...)
		// erase by given index
		.def("__delitem__", &erase_idx<node_type>, "idx"_a)
		.def("erase",       &erase_idx<node_type>, "idx"_a, "Erase link with given index")
		// erase given link
		.def("__delitem__", &erase_link<node_type>, "link"_a, gil...)
		.def("erase",       &erase_link<node_type>, "link"_a, "Erase given link", gil...)
		;
	};

	///////////////////////////////////////////////////////////////////////////////
	//  bare_node
	//
	auto bare_node_pyface = py::class_<bare_node>(m, "bare_node")
		.def(py::init<const node&>())

		.def("__iter__", [](const bare_node& N) {
			return make_container_iterator(N.leafs());
		})

		// convert to node
		.def("armed", &bare_node::armed, "Convert to safe node")

		// unsafe insert link
		.def("insert", py::overload_cast<unsafe_t, link, InsertPolicy>(&bare_node::insert),
			"unsafe"_a, "link"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert given link (unsafe for link)")
		// unsafe insert bunch of links
		.def("insert", py::overload_cast<unsafe_t, links_v, InsertPolicy>(&bare_node::insert),
			"unsafe"_a, "link"_a, "pol"_a = InsertPolicy::AllowDupNames,
			"Insert bunch of links (unsafe for links)")

		.def("clear", &bare_node::clear, "Clears all node contents")

		.def("rearrange", py::overload_cast<lids_v>(&bare_node::rearrange),
			"new_order"_a, "Apply custom order to node")
		.def("rearrange", py::overload_cast<std::vector<std::size_t>>(&bare_node::rearrange),
			"new_order"_a, "Apply custom order to node")
	;

	add_common_api(bare_node_pyface);

	///////////////////////////////////////////////////////////////////////////////
	//  node
	//
	// [NOTE] overloads binding order is essential, so bind common API first
	auto node_pyface = py::class_<node, engine>(m, "node")
		.def(py::init<>())
		.def(py::init<const bare_node&>())
	;
	add_common_api(node_pyface);

	// append node-specific API
	node_pyface
		.def("__iter__", [](const node& N) {
			return make_container_iterator(N.leafs());
		})

		.def_property_readonly_static("nil", [](const py::object&) { return node::nil(); })

		.def("bare", &node::bare, "Get bare (unsafe) node")

		.def("clone", &node::clone, "deep"_a = false, "Make shallow or deep copy of node")

		.def("skeys", &node::skeys, "key_meaning"_a, "ordering"_a = Key::AnyOrder, nogil)

		// check if node contains key
		// ... by object
		.def("__contains__", &contains_obj, "obj"_a)
		.def("contains",     &contains_obj, "obj"_a, "Check if node contains link to given object")
		// ... generic version
		.def("contains", &contains_key, "key"_a, "key_meaning"_a, "Check if node contains link with given key")

		// search by object instance
		.def("__getitem__", &find_obj<true>, "obj"_a)
		.def("find",        &find_obj<false>, "obj"_a, "Find link to given object")
		// search by string key & key treatment
		.def("find", py::overload_cast<std::string, Key>(&node::find, py::const_),
			"key"_a, "key_meaning"_a, "Find link by key with specified treatment", nogil)

		// obtain index in custom order from object
		.def("index", &index_obj, "obj"_a, "Find index of link to given object")
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

		// insert link at given index
		.def("insert", [](node& N, link l, const long idx, InsertPolicy pol) {
			return N.insert(std::move(l), idx, pol);
		}, "link"_a, "idx"_a, "pol"_a = InsertPolicy::AllowDupNames, "Insert link at given index")

		// erase by object instance
		.def("__delitem__", &erase_obj, "obj"_a)
		.def("erase",       &erase_obj, "obj"_a, "Erase links to given object")
		// generic erase by key & meaning
		.def("erase", py::overload_cast<std::string, Key>(&node::erase, py::const_),
			"key"_a, "key_meaning"_a = Key::Name, "Erase all leafs with given key", nogil)

		// misc container-related functions
		.def("clear", py::overload_cast<>(&node::clear, py::const_), "Clears all node contents")
		.def("clear", py::overload_cast<launch_async_t>(&node::clear, py::const_),
			"Async clear all node contents")

		// link rename
		.def("rename", py::overload_cast<std::string, std::string>(&node::rename, py::const_),
			"old_name"_a, "new_name"_a, "Rename all links with given old_name")
		// by index offset
		.def("rename", py::overload_cast<std::size_t, std::string>(&node::rename, py::const_),
			"idx"_a, "new_name"_a, "Rename link with given index")
		// by ID
		.def("rename", py::overload_cast<lid_type, std::string>(&node::rename, py::const_),
			"lid"_a, "new_name"_a, "Rename link with given ID")

		.def("rearrange", py::overload_cast<lids_v>(&node::rearrange, py::const_),
			"new_order"_a, "Apply custom order to node")
		.def("rearrange", py::overload_cast<std::vector<std::size_t>>(&node::rearrange, py::const_),
			"new_order"_a, "Apply custom order to node")

		// events subscrition
		.def("subscribe", [](const node& N, node::event_handler f, Event listen_to) {
			return N.subscribe(pipe_through_queue(std::move(f), launch_async), listen_to);
		}, "event_cb"_a, "events"_a = Event::All)

		.def("unsubscribe", py::overload_cast<deep_t>(&node::unsubscribe, py::const_))
		.def("unsubscribe", py::overload_cast<>(&engine::unsubscribe, py::const_))
		// [NOTE] need to define non-static method for overloading
		.def("unsubscribe", [](const py::object&, std::uint64_t handler_id) {
			engine::unsubscribe(handler_id);
		}, "handler_id"_a)

		.def("apply",
			[](const node& N, std::function< py::object(bare_node) > tr) {
				N.apply(launch_async, pytr_through_queue(std::move(tr)));
			},
			"tr"_a, "Send transaction `tr` to node's queue, return immediately"
		)
	;

	// node::weak_ptr
	bind_weak_ptr(node_pyface);
}

NAMESPACE_END(blue_sky::python)
