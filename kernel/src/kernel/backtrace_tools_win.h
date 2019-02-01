/// @file
/// @author Alexander Gagarin (uentity)
/// @date 04.04.2018
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/error.h>
#include <bs/log.h>
#include <windows.h>
#include <dbghelp.h>
#include <vector>
#include <string>
#include <spdlog/fmt/fmt.h>

#pragma comment (lib, "dbghelp.lib")

namespace {

int sys_get_backtrace(void **backtrace_, int size_) {
	return CaptureStackBackTrace(0, size_, backtrace_, NULL);
}

char** sys_get_backtrace_names(void *const *backtrace_, int size_) {
	char symbol_ [sizeof (IMAGEHLP_SYMBOL64) + sizeof (TCHAR) * (MAX_PATH + 1)] = {0};
	IMAGEHLP_SYMBOL64 *symbol = (IMAGEHLP_SYMBOL64 *)symbol_;

	symbol->SizeOfStruct = sizeof (IMAGEHLP_SYMBOL64);
	symbol->MaxNameLength = MAX_PATH;

	HANDLE process = GetCurrentProcess ();
	char **names = (char **)malloc ((MAX_PATH + 1 + sizeof (char *)) * size_);
	memset (names, 0, (MAX_PATH + 1 + sizeof (char *)) * size_);

	for(int i = 0; i < size_; ++i) {
		names[i] = (char *)names + sizeof (char *) * size_ + (MAX_PATH + 1) * i;

		BOOL res = SymGetSymFromAddr64 (process, (DWORD64)backtrace_[i], 0, symbol);
		if(!res) {
			LPVOID lpMsgBuf;
			DWORD dw = GetLastError(); 

			FormatMessage(
				FORMAT_MESSAGE_ALLOCATE_BUFFER | 
				FORMAT_MESSAGE_FROM_SYSTEM |
				FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL,
				dw,
				MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
				(LPTSTR) &lpMsgBuf,
				0, NULL );

			BSERROR << (char*)lpMsgBuf << bs_end;
			LocalFree (lpMsgBuf);

			break;
		}

		memcpy (names[i], symbol->Name, (std::min <size_t>) (MAX_PATH, strlen (symbol->Name)));
	}

	return names;
}

std::vector<std::string> sys_demangled_backtrace_names(void** callstack, char** symbollist, int size, int skip = 1) {
	// result strings will go here
	std::vector<std::string> res;

	for (int i = skip; i < size; i++) {
		res.emplace_back(symbollist[i]);
	}

	free(symbollist);
	return res;
}

} // hidden namespace

