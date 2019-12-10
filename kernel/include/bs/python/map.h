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

template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
auto bind_reach_map(py::handle scope, std::string const &name, Args&&... args) {
	using KeyType = typename Map::key_type;
	using MappedType = typename Map::mapped_type;

	// first bindings provided by pybind11
	auto cl = pybind11::bind_map<Map>(scope, name, std::forward<Args>(args)...);

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

NAMESPACE_END(blue_sky::python)
