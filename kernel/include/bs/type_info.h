/// @file
/// @author uentity
/// @date 28.04.2016
/// @brief BlueSky type info class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "setup_common_api.h"
#include <typeinfo>
#include <typeindex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

#define BS_TYPE_INFO std::type_index
#define BS_GET_TI(T) (BS_TYPE_INFO(typeid(T)))

namespace blue_sky { namespace detail {
/// Following code based on similar chunk from pybind11

/// Erase all occurrences of a substring
inline void erase_all(std::string &string, const std::string &search) {
	for (size_t pos = 0;;) {
		pos = string.find(search, pos);
		if (pos == std::string::npos) break;
		string.erase(pos, search.length());
	}
}

inline void clean_type_id(std::string &name) {
#if defined(__GNUG__)
	int status = 0;
	std::unique_ptr<char, void (*)(void *)> res {
		abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free };
	if (status == 0)
		name = res.get();
#else
	erase_all(name, "class ");
	erase_all(name, "struct ");
	erase_all(name, "enum ");
#endif
	erase_all(name, "blue_sky::");
}

} // eof blue_sky::detail namespace

// dumb typedef for compatibility reasons - to be removed
typedef std::type_index bs_type_info;

// empty class denotes "Nil" type - associated with nothing
BS_API const std::type_index& nil_type_info();
BS_API const std::string& nil_type_name();

// check if type is nil
BS_API bool is_nil(const std::type_index&);

/// Return a string representation of a C++ type
template <typename T> static std::string bs_type_name() {
	std::string name(typeid(T).name());
	detail::clean_type_id(name);
	return name;
}

} // eof blue_sky namespace

