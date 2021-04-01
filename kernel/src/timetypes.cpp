/// @file
/// @author uentity
/// @date 19.03.2019
/// @brief Implementation details for time-related types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/common.h>
#include <bs/timetypes.h>

#include <fmt/format.h>
#include <fmt/chrono.h>

#include <ctime>

NAMESPACE_BEGIN(blue_sky)
namespace C = std::chrono;
using namespace std::chrono_literals;

timestamp make_timestamp() {
	return caf::make_timestamp();
}

std::string to_string(timespan t) {
	std::string res;
	auto fmt_and_cut = [&res, &t](auto part) {
		if(part.count() > 0) {
			if(!res.empty()) res.push_back(' ');
			fmt::format_to(std::back_inserter(res), "{}", part);
			t -= part;
		}
	};

	fmt_and_cut(C::floor<C::hours>(t));
	fmt_and_cut(C::floor<C::minutes>(t));
	fmt_and_cut(C::floor<C::seconds>(t));
	fmt_and_cut(C::floor<C::milliseconds>(t));
	fmt_and_cut(C::floor<C::microseconds>(t));
	fmt_and_cut(C::floor<C::nanoseconds>(t));
	if(res.empty())
		fmt::format_to(std::back_inserter(res), "{}", 0ns);
	return res;
}

std::string to_string(timestamp t) {
	using sys_duration = C::system_clock::duration;
	// write down date and time
	auto sys_t = C::system_clock::to_time_t(C::time_point_cast<sys_duration>(t));
	std::string res;
	fmt::format_to(std::back_inserter(res), "{:%FT%T}", *std::localtime(&sys_t));
	// write residual ms
	auto residual_ms = C::duration_cast<C::milliseconds>(t - C::system_clock::from_time_t(sys_t));
	if(residual_ms.count() > 0)
		fmt::format_to(std::back_inserter(res), ".{}", residual_ms.count());
	return res;
}

std::ostream& operator <<(std::ostream& os, timestamp t) {
	return os << to_string(t), os;
}

std::ostream& operator <<(std::ostream& os, timespan t) {
	return os << to_string(t), os;
}

NAMESPACE_END(blue_sky)
