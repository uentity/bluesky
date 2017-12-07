/// @file
/// @author uentity, NikonovMA a ka no_NaMe <__no_name__@rambler.ru>
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/misc.h>
#include <stdarg.h>
#include <cstring>

#ifdef UNIX
#include <errno.h>
#include <dlfcn.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include "Shlwapi.h"
#include <time.h>
#endif

namespace blue_sky {

//! \return string of chars, contains current time
std::string gettime() {
	time_t cur_time;
	char * cur_time_str = nullptr;
	if(time(&cur_time)) {
		cur_time_str = ctime(&cur_time);
		if(cur_time_str)
			cur_time_str[strlen(cur_time_str)-1] = '\0';
	}
	return cur_time_str;
}

// using code from from boos/filesystem/src/exception.cpp
// system error-messages
std::string system_message(int err_code) {
	std::string str;
#ifdef _WIN32
	LPSTR lpMsgBuf;
	::FormatMessageA(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		err_code,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
		(LPSTR)&lpMsgBuf,
		0,
		NULL
	);
	str = static_cast< const char* >(lpMsgBuf);
	::LocalFree( lpMsgBuf ); // free the buffer
	str = str.substr(0, str.find('\n'));
	str = str.substr(0, str.find('\r'));
#else
	str = ::strerror(err_code);
#endif
	return str;
}

std::string last_system_message() {
	int err_code;
#ifdef _WIN32
	err_code = GetLastError();
#else
	err_code = errno;
#endif
	return system_message(err_code);
}

BS_API bool is_path_local(const std::string& path) {
#ifdef _WIN32
	// first check if given path is valid and exists
	const auto ppath = path.c_str();
	if (!PathFileExists(ppath))
		return false;
	// obtain volume name (drive letter)
	std::vector<char> vol_name(MAX_PATH + 1, 0);
	if(!GetVolumePathName(ppath, &vol_name[0], (DWORD)vol_name.size()))
		return false;
	// return true for local drives
	return (GetDriveType(&vol_name[0]) == DRIVE_FIXED);
#else
	// for Linux always return true
	return true;
#endif
}

}	//end of namespace blue_sky

