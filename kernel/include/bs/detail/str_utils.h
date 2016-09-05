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
#include <string>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

namespace blue_sky {

//! this manipulator skips line
template <class charT, class traits>
inline
std::basic_istream <charT, traits>&
ignoreLine(std::basic_istream <charT, traits>& strm)
{
	strm.ignore(0x7fff, strm.widen('\n'));
	return strm;
}

// trim from start
static inline std::string ltrim(const std::string& S) {
	std::string s = S;
	s.erase(s.begin(), std::find_if(
		s.begin(), s.end(),
		std::not1(std::ptr_fun<int, int>(std::isspace))
	));
	return s;
}

// trim from end
static inline std::string rtrim(const std::string& S) {
	std::string s = S;
	s.erase(
		std::find_if(
			s.rbegin(), s.rend(),
			std::not1(std::ptr_fun<int, int>(std::isspace))
		).base(),
		s.end()
	);
	return s;
}

// trim from both ends
static inline std::string trim(const std::string& s) {
	return ltrim(rtrim(s));
}

// functions to convert string <-> wstring using given locale name in POSIX format
// if passed enc_name is empty, then native system locale is auto-deduced
// if enc_name = "UTF-8" or "utf-8", then country-based UTF-8 locale is used
BS_API std::string wstr2str(const std::wstring& text, const std::string& loc_name = "utf-8");
BS_API std::wstring str2wstr(const std::string& text, const std::string& loc_name = "utf-8");

// convert UTF-8 encoded string <-> string
// if passed enc_name is empty, then native system locale is auto-deduced for string
BS_API std::string ustr2str(const std::string& text, const std::string& loc_name = "utf-8");
BS_API std::string str2ustr(const std::string& text, const std::string& loc_name = "utf-8");

// generic string -> string conversion from one locale to another
// if in_enc_name is empty, assume that text given in native system locale
BS_API std::string str2str(
	const std::string& text, const std::string& out_loc_name,
	const std::string& in_loc_name = ""
);

}	//end of blue_sky namespace

