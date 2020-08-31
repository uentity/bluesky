/// @file
/// @author uentity
/// @date 19.03.2019
/// @brief Property implementation details
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/property.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

NAMESPACE_BEGIN(blue_sky::prop)

template<typename T>
auto printv(std::ostream& os, const T& v) {
	// surround strings with double quotes
	if constexpr(std::is_same_v<T, string>) os << '"';
	// specific processing of time types
	if constexpr(std::is_same_v<T, timespan> || std::is_same_v<T, timestamp>)
		os << blue_sky::to_string(v);
	else if constexpr(std::is_same_v<T, boolean>)
		os << (v ? std::string_view("true") : std::string_view("false"));
	else os << v;
	if constexpr(std::is_same_v<T, string>) os << '"';
}

template<typename T>
auto& operator <<(std::ostream& os, const std::vector<T>& x) {
	os << '[';
	bool first = true;
	for(auto&& v : x) {
		if(first) first = false;
		else os << ", ";
		printv(os, v);
	}
	return os << ']', os;
}

auto operator <<(std::ostream& os, const property& x) -> std::ostream& {
	return visit([&os](auto&& v){
		if(!is_none(v))
			printv(os, v);
		else
			os << "None";
	}, x), os;
}

auto to_string(const property& p) -> std::string {
	return fmt::format("{}", p);
}

NAMESPACE_END(blue_sky::prop)
