/// @file
/// @author uentity
/// @date 23.05.2019
/// @brief Extend opaque map-like container bindings provided by pybind11
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"

NAMESPACE_BEGIN(blue_sky::python)
NAMESPACE_BEGIN(detail)

// trait to detect whether map has `erase()` method or not
template<typename Map, typename = void> struct has_erase : std::false_type {};

template<typename Map>
struct has_erase<Map, std::void_t<decltype(std::declval<Map>().erase(typename Map::iterator{}))>> :
	std::true_type {};

template<typename Map>
inline constexpr auto has_erase_v = has_erase<Map>::value;

// gen base bindings for map without `erase()`
template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
auto bind_growing_map(py::handle scope, const std::string &name, Args&&... args) {
	using KeyType = typename Map::key_type;
	using MappedType = typename Map::mapped_type;
	using Class_ = py::class_<Map, holder_type>;
	namespace detail = py::detail;
	using namespace py;

	// If either type is a non-module-local bound type then make the map binding non-local as well;
	// otherwise (e.g. both types are either module-local or converting) the map will be
	// module-local.
	auto tinfo = detail::get_type_info(typeid(MappedType));
	bool local = !tinfo || tinfo->module_local;
	if (local) {
		tinfo = detail::get_type_info(typeid(KeyType));
		local = !tinfo || tinfo->module_local;
	}

	Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

	cl.def(init<>());

	// Register stream insertion operator (if possible)
	detail::map_if_insertion_operator<Map, Class_>(cl, name);

	cl.def("__bool__",
		[](const Map &m) -> bool { return !m.empty(); },
		"Check whether the dict is nonempty"
	);

	cl.def("__iter__",
		[](Map &m) { return make_key_iterator(m.begin(), m.end()); },
		keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
	);

	cl.def("items",
		[](Map &m) { return make_iterator(m.begin(), m.end()); },
		keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
	);

	cl.def("__getitem__",
		[](Map &m, const KeyType &k) -> MappedType & {
			auto it = m.find(k);
			if (it == m.end())
				throw key_error();
			return it->second;
		},
		return_value_policy::reference_internal // ref + keepalive
	);

	cl.def("__contains__",
		[](Map &m, const KeyType &k) -> bool {
			auto it = m.find(k);
			if (it == m.end())
				return false;
			return true;
		}
	);

	// Assignment provided only if the type is copyable
	detail::map_assignment<Map, Class_>(cl);

	cl.def("__len__", &Map::size);

	return cl;
}

// generate base binding dependith on whether `erase()` presents or not
template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
auto bind_base_map(py::handle scope, const std::string &name, Args&&... args) {
	if constexpr(has_erase_v<Map>)
		return py::bind_map<Map>(scope, name, std::forward<Args>(args)...);
	else
		return bind_growing_map<Map>(scope, name, std::forward<Args>(args)...);
};

// additional methods for map-like containers
template<typename Map, typename PyMap>
auto make_rich_map(PyMap& cl) -> PyMap& {
	using KeyType = typename Map::key_type;
	using MappedType = typename Map::mapped_type;

	// allow constructing from Python dict
	const auto from_pydict = [](Map& tgt, py::dict src) {
		auto key_caster = py::detail::make_caster<KeyType>();
		auto value_caster = py::detail::make_caster<MappedType>();
		for(const auto& [sk, sv] : src) {
			if(!key_caster.load(sk, true) || !value_caster.load(sv, true)) continue;
			tgt[py::detail::cast_op<KeyType>(key_caster)] = py::detail::cast_op<MappedType>(value_caster);
		}
	};
	cl.def(py::init([&from_pydict](py::dict D) {
		Map M;
		from_pydict(M, std::move(D));
		return M;
	}));
	// and explicit conversion from Py dict
	cl.def("from_dict", from_pydict, "src"_a);

	// copy content to Python dict
	cl.def("to_dict", [](const Map& m) {
		py::dict res;
		for(const auto& [k, v] : m)
			res[py::cast(k)] = py::cast(v);
		return res;
	});

	cl.def("keys", [](const Map& m) {
		py::list res;
		for(const auto& [k, v] : m)
			res.append(k);
		return res;
	});

	cl.def("values", [](const Map& m) {
		py::list res;
		for(const auto& [k, v] : m)
			res.append(v);
		return res;
	});

	const auto contains = [](const Map& m, py::object key) {
		auto key_caster = py::detail::make_caster<KeyType>();
		if(key_caster.load(key, true))
			return m.find(py::detail::cast_op<KeyType>(key_caster)) != m.end();
		return false;
	};
	cl.def("__contains__", contains);
	cl.def("has_key", contains, "key"_a);

	cl.def("clear", [](Map& m) { m.clear(); });

	if constexpr(has_erase_v<Map>)
		cl.def("pop", [](Map& m, py::object key, py::object def_value) {
			auto key_caster = py::detail::make_caster<KeyType>();
			if(!key_caster.load(key, true)) return def_value;

			auto pval = m.find(py::detail::cast_op<KeyType>(key_caster));
			if(pval != m.end()) {
				auto res = py::cast(pval->second);
				m.erase(pval);
				return res;
			}
			return def_value;
		}, "key"_a, "default"_a = py::none());

	cl.def("get", [](Map& m, py::object key, py::object def_value) {
		auto key_caster = py::detail::make_caster<KeyType>();
		if(!key_caster.load(key, true)) return def_value;

		auto pval = m.find(py::detail::cast_op<KeyType>(key_caster));
		return pval != m.end() ? py::cast(pval->second) : std::move(def_value);
	}, "key"_a, "default"_a = py::none());

	cl.def(py::self == py::self);
	cl.def(py::self != py::self);
	cl.def(py::self < py::self);
	cl.def(py::self <= py::self);
	cl.def(py::self > py::self);
	cl.def(py::self >= py::self);

	// make Python dictionary implicitly convertible to Map
	py::implicitly_convertible<py::dict, Map>();
	return cl;
}

NAMESPACE_END(detail)

/// rich binder for map-like container (autodetects `erase()` presence)
template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
auto bind_rich_map(py::handle scope, std::string const &name, Args&&... args) {
	// gen base bindings
	auto cl = detail::bind_base_map<Map>(scope, name, std::forward<Args>(args)...);
	// add rich methods
	detail::make_rich_map<Map>(cl);
	return cl;
}

NAMESPACE_END(blue_sky::python)
