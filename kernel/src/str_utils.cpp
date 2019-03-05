/// @file
/// @author uentity
/// @date 05.08.2016
/// @brief String utils implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/detail/str_utils.h>
#include <boost/locale.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace blue_sky {

// hidden namespace
namespace {
// BS kernel locale generator
struct loc_storage {
	loc_storage()
		: native_loc(boost::locale::util::get_system_locale())
	{
		gloc.locale_cache_enabled(true);
		const std::locale& native = gloc.generate(native_loc);
		native_loc_prefix =
			std::use_facet< boost::locale::info >(native).language() + "_" +
			std::use_facet< boost::locale::info >(native).country();
		native_loc_utf8 = native_loc_prefix + ".UTF-8";

#ifdef _WIN32
		// boost's get_system_locale() is affected by environment variables and
		// sometimes fails to discover valid _system_ encoding
		// but it is able to corretly determine regional settings
		//
		// find active Windows codepage
		ulong acp = ulong(GetACP());
		if(acp == 65001) {
			// Windows codepage is UTF-8
			native_loc = native_loc_utf8;
		}
		else {
			native_loc = native_loc_prefix + ".CP" + boost::lexical_cast< std::string >(acp);
		}
#endif
		// DEBUG
		//BSOUT << "Native locale: " << native_loc << ", " << native_loc_utf8 << bs_end;
	}

	// obtain locale
	// if empty locale passed, generate native system locale
	std::locale operator()(const std::string& loc_name = "") const {
		if(loc_name.empty())
			return gloc.generate(native_loc);
		else if(loc_name == "utf-8" || loc_name == "UTF-8")
			return gloc.generate(native_loc_utf8);
		else
			return gloc.generate(loc_name);
	}

	// return UTF-8 locale with native system country settings
	std::locale native_utf8() const {
		return operator()(native_loc_utf8);
	}

	boost::locale::generator gloc;
	std::string native_loc;
	std::string native_loc_prefix;
	std::string native_loc_utf8;
};
// storage singleton
static loc_storage ls_;

} // eof hidden namespace

// functions to convert string <-> wstring
std::string wstr2str(const std::wstring& text, const std::string& loc_name) {
	return boost::locale::conv::from_utf(text, ls_(loc_name));
}

std::wstring str2wstr(const std::string& text, const std::string& loc_name) {
	return boost::locale::conv::to_utf< wchar_t >(text, ls_(loc_name));
}

std::string ustr2str(const std::string& text, const std::string& loc_name) {
	return boost::locale::conv::from_utf(text, ls_(loc_name));
}

std::string str2ustr(const std::string& text, const std::string& loc_name) {
	return boost::locale::conv::to_utf< char >(text, ls_(loc_name));
}

std::string str2str(
	const std::string& text, const std::string& out_loc_name, const std::string& in_loc_name
) {
	if(in_loc_name.size())
		return boost::locale::conv::between(text, out_loc_name, in_loc_name);
	else
		return boost::locale::conv::between(text, out_loc_name, ls_.native_loc);
}

std::locale get_locale(const std::string& loc_name) {
	return ls_(loc_name);
}

}	//end of namespace blue_sky

