/// @file
/// @author Alexander Gagarin (uentity)
/// @date 03.04.2018
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once
#include <execinfo.h> // for backtrace
#include <dlfcn.h>    // for dladdr
#include <cxxabi.h>   // for __cxa_demangle

#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <spdlog/fmt/fmt.h>

namespace {

int sys_get_backtrace(void **backtrace_, int size_) {
	return backtrace (backtrace_, size_);
}

char** sys_get_backtrace_names(void *const *backtrace_, int size_) {
	return backtrace_symbols (backtrace_, size_);
}

#if 1
std::vector<std::string> sys_demangled_backtrace_names(void** callstack, char** symbollist, int size, int skip = 1) {
	// result strings will go here
	std::vector<std::string> res;

	// allocate string which will be filled with the demangled function name
	size_t funcnamesize = 1024;
	char* funcname = (char*)malloc(funcnamesize);

	// iterate over the returned symbol lines. skip the first, it is the
	// address of this function.
	for (int i = skip; i < size; i++) {
		char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

		// find parentheses and +address offset surrounding the mangled name:
		// ./module(function+0x15c) [0x8048a6d]
		for (char *p = symbollist[i]; *p; ++p) {
			if (*p == '(')
				begin_name = p;
			else if (*p == '+')
				begin_offset = p;
			else if (*p == ')' && begin_offset) {
				end_offset = p;
				break;
			}
		}

		if(begin_name && begin_offset && end_offset && begin_name < begin_offset) {
			*begin_name++ = '\0';
			*begin_offset++ = '\0';
			*end_offset = '\0';

			// mangled name is now in [begin_name, begin_offset) and caller
			// offset in [begin_offset, end_offset). now apply
			// __cxa_demangle():

			int status;
			char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
			if (status == 0) {
				funcname = ret; // use possibly realloc()-ed string
				res.emplace_back(fmt::format("{} : {} +{}", symbollist[i], funcname, begin_offset));
			}
			else {
				// demangling failed. Output function name as a C function with
				// no arguments.
				res.emplace_back(fmt::format("{} : {} +{}", symbollist[i], begin_name, begin_offset));
			}
		}
		else {
			// couldn't parse the line? print the whole line.
			res.emplace_back(symbollist[i]);
		}
	}

	free(funcname);
	free(symbollist);
	return res;
}

#else
std::vector<std::string> sys_demangled_backtrace_names(void** callstack, char** symbols, int nFrames, int skip = 1) {
	// result strings will go here
	std::vector<std::string> res;

	char buf[1024];

	for (int i = skip; i < nFrames; i++) {
		Dl_info info;
		if (dladdr(callstack[i], &info) && info.dli_sname) {
			char *demangled = NULL;
			int status = -1;
			if (info.dli_sname[0] == '_')
				demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
			snprintf(buf, sizeof(buf), "%-3d %*p %s + %zd",
					i, int(2 + sizeof(void*) * 2), callstack[i],
					status == 0 ? demangled :
					info.dli_sname == 0 ? symbols[i] : info.dli_sname,
					(char *)callstack[i] - (char *)info.dli_saddr);
			free(demangled);
		} else {
			snprintf(buf, sizeof(buf), "%-3d %*p %s",
					i, int(2 + sizeof(void*) * 2), callstack[i], symbols[i]);
		}
		res.push_back(buf);
	}

	free(symbols);
	return res;
}
#endif

} // hidden namespace

