/// @file
/// @author uentity
/// @date 05.08.2016
/// @brief String utils - locale conversion, etc
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>

#include <algorithm>
#include <cctype>
#include <locale>

NAMESPACE_BEGIN(blue_sky)

/// manipulator that skips line
template<class charT, class traits>
auto ignoreLine(std::basic_istream <charT, traits>& strm) -> std::basic_istream <charT, traits>& {
	strm.ignore(0x7fff, strm.widen('\n'));
	return strm;
}

/// trim from start
template<typename String>
auto ltrim(String&& S) {
	std::string s = std::forward<String>(S);
	s.erase(
		s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !std::isspace(ch); })
	);
	return s;
}

/// trim from end
template<typename String>
auto rtrim(String&& S) {
	std::string s = std::forward<String>(S);
	s.erase(
		std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(),
		s.end()
	);
	return s;
}

/// trim from both ends
template<typename String>
auto trim(String&& S) {
	return ltrim(rtrim(std::forward<String>(S)));
}

// functions to convert string <-> wstring using given locale name in POSIX format
// if passed enc_name is empty, then native system locale is auto-deduced
// if enc_name = "UTF-8" or "utf-8", then country-based UTF-8 locale is used
BS_API std::string wstr2str(const std::wstring& text, const std::string& loc_name = "");
BS_API std::wstring str2wstr(const std::string& text, const std::string& loc_name = "");

// convert UTF-8 encoded string <-> string
// if passed enc_name is empty, then native system locale is auto-deduced for string
BS_API std::string ustr2str(const std::string& text, const std::string& loc_name = "");
BS_API std::string str2ustr(const std::string& text, const std::string& loc_name = "");

// generic string -> string conversion from one locale to another
// if in_enc_name is empty, assume that text given in native system locale
BS_API std::string str2str(
	const std::string& text, const std::string& out_loc_name,
	const std::string& in_loc_name = ""
);

// obtain std::locale for given locale string name
BS_API std::locale get_locale(const std::string& loc_name = "");

NAMESPACE_END(blue_sky)
