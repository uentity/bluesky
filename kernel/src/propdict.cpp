/// @file
/// @author uentity
/// @date 21.03.2019
/// @brief Properties dictionary implementation details
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/propdict.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

NAMESPACE_BEGIN(blue_sky::prop)

std::ostream& operator <<(std::ostream& os, const propdict& x) {
	os << '{';
	bool first = true;
	for(auto&& v : x) {
		if(first) first = false;
		else os << ", ";
		os << '{' << v.first << ": " << v.second << '}';
	}
	return os << '}', os;
}

std::string to_string(const propdict& p) {
	return fmt::format("{}", p);
}

NAMESPACE_END(blue_sky::prop)
