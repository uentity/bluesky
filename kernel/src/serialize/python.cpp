/// @file
/// @author uentity
/// @date 08.11.2018
/// @brief Python types serialization using pickle
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/python.h>

namespace py = pybind11;

/*-----------------------------------------------------------------------------
 *  hiddent serialization impl
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()
using namespace cereal;

auto dumps(const py::object& obj) -> std::string {
	static const py::handle dumps = py::module::import("pickle").attr("dumps");
	return py::cast<std::string>(dumps(obj, -1));
}

auto loads(const std::string& obj_dump) -> py::object {
	static const py::handle loads = py::module::import("pickle").attr("loads");
	return loads(py::bytes(obj_dump));
}

///////////////////////////////////////////////////////////////////////////////
//  save path
//
// text archives
template<typename Archive, traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
auto save(Archive& ar, py::object const& t) -> void {
	auto obj_dump = dumps(t);
	ar(make_nvp("size", obj_dump.size()));
	ar.saveBinaryValue(obj_dump.data(), sizeof(std::string::value_type)*obj_dump.size(), "data");
}
// binary archives
template<typename Archive, traits::DisableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
auto save(Archive& ar, py::object const& t) -> void {
	ar(dumps(t));
}

///////////////////////////////////////////////////////////////////////////////
//  load path
//
// text archives
template<typename Archive, traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
auto load(Archive& ar, py::object& t) -> void {
	// load dump size
	std::size_t sz;
	ar(make_nvp("size", sz));
	// decode dumped state
	std::string obj_dump(sz, ' ');
	ar.loadBinaryValue(&obj_dump[0], sizeof(std::string::value_type)*obj_dump.size(), "data");
	// construct Python object
	t = loads(obj_dump);
}
// binary archives
template<typename Archive, traits::DisableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
auto load(Archive& ar, py::object& t) -> void {
	// load object state
	std::string obj_dump;
	ar(obj_dump);
	// construct Python object
	t = loads(obj_dump);
}

///////////////////////////////////////////////////////////////////////////////
//  split serialize into save/load
//
// save
template<typename Archive>
auto serialize_(Archive& ar, py::object const& t, std::true_type /* saving */) -> void {
	save(ar, t);
}
// load
template<typename Archive>
auto serialize_(Archive& ar, py::object& t, std::false_type /* saving */) -> void {
	load(ar, t);
}

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  py::handle serialize impl
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, py::object)
	using is_saving = typename Archive::is_saving;
	serialize_(ar, t, is_saving());
BSS_FCN_END

BSS_FCN_EXPORT(serialize, py::object)

