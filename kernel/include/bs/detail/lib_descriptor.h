// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/common.h>

// hide implementation
namespace blue_sky { namespace detail {
/*-----------------------------------------------------------------------------
 *  Shared library descriptor
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API lib_descriptor {
	std::string fname_; //!< path to dynamic library
	// seems to be the same in both UNIX and Windows
	void* handle_;

	lib_descriptor();

	bool load(const char* fname);

	void unload();

	template< typename fn_t >
	bool load_sym(const char* sym_name, fn_t& fn) const {
		fn = (fn_t)load_sym_name(sym_name);
		return bool(fn);
	}

	template< class sym_t >
	static int load_sym_glob(const char* sym_name, sym_t& sym) {
		sym = (sym_t)load_sym_glob_name(sym_name);
		// NOTE: return 0 on success
		return int(sym == nullptr);
	}

	static std::string dll_error_message();

	// lib_descriptors are comparable by file name
	friend bool operator <(const lib_descriptor& lhs, const lib_descriptor& rhs);

	// equality operator for lib's handlers
	friend bool operator ==(const lib_descriptor& left, const lib_descriptor& right);

private:
	void* load_sym_name(const char* sym_name) const;
	static void* load_sym_glob_name(const char* sym_name);
};

}} // eof blue_sky::detail namespace

