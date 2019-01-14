/// @file
/// @author uentity
/// @date 01.09.2016
/// @brief lib_descriptor implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/detail/lib_descriptor.h>
#include <bs/plugin_descriptor.h>
#include <bs/error.h>
#include <bs/misc.h>
#include <bs/kernel/errors.h>

#ifdef _WIN32
#include <Psapi.h>
#elif defined(UNIX)
#include <dlfcn.h>
#endif

namespace blue_sky { namespace detail {


lib_descriptor::lib_descriptor()
	: fname_(""), handle_(nullptr)
{}

bool lib_descriptor::load(const char* fname) {
	fname_ = fname;
#ifdef UNIX
	handle_ = dlopen(fname, RTLD_GLOBAL | RTLD_NOW);
#else
	handle_ = LoadLibrary(fname);
#endif
	if (!handle_) {
		throw error(
			std::string("lib_descriptor: ") + fname + ": " + dll_error_message(),
			kernel::Error::CantLoadDLL
		);
	}
	return (bool)handle_;
}

void lib_descriptor::unload() {
	// unload library
	
	if(handle_) {
		bool res = false;
#ifdef UNIX
		res = dlclose(handle_) == 0;
#elif defined(_WIN32)
		res = FreeLibrary(handle_);
#endif
		if(!res) {
			throw error(
				std::string("lib_descriptor: ") + fname_ + ": " + dll_error_message(),
				kernel::Error::CantUnloadDLL
			);
		}
	}
	handle_ = nullptr;
}

void* lib_descriptor::load_sym_name(const char* sym_name) const {
	if(handle_) {
#ifdef UNIX
		return (void*)dlsym(handle_, sym_name);
#else
		return (void*)GetProcAddress(handle_, LPCSTR(sym_name));
#endif
	}
	return nullptr;
}

void* lib_descriptor::load_sym_glob_name(const char* sym_name) {
#ifdef UNIX
	return (void*)dlsym(RTLD_DEFAULT, sym_name);
#elif defined(_WIN32)
	//helper struct to find BlueSky kernel among all loaded modules
	struct find_kernel_module {
		static HMODULE go() {
			HANDLE hproc;
			HMODULE hmods[1024];
			DWORD cbNeeded;
			BS_GET_PLUGIN_DESCRIPTOR pdf = nullptr;
			plugin_descriptor* pd = nullptr;
			ulong m_ind = 0;

			// get handle of current process
			hproc = GetCurrentProcess();

			// enumerate all modules of current process
			HMODULE res = nullptr;
			if(hproc && EnumProcessModules(hproc, hmods, sizeof(hmods), &cbNeeded))	{
				ulong cnt = cbNeeded / sizeof(HMODULE);
				for (ulong i = 0; i < cnt; ++i) {
					// search for given symbol in i-th module
					if(pdf = (BS_GET_PLUGIN_DESCRIPTOR)GetProcAddress(hmods[i], "bs_get_plugin_descriptor")) {
						// get pointer to plugin_descriptor & check if this is a kernel
						if((pd = pdf()) && pd->name.compare("kernel") == 0) {
							res = hmods[i];
							break;
						}
					}
				}
				CloseHandle(hproc);
			}
			return res;
		}
	};

	//get kernel module handle
	static HMODULE km = find_kernel_module::go();

	//search for given symbol
	if(km)
		return (void*)GetProcAddress(km, sym_name);
	return nullptr;
#endif
}

std::string lib_descriptor::dll_error_message() {
#ifdef UNIX
	return dlerror();
#else
	return last_system_message();
#endif
}

// lib_descriptors are comparable by file name
bool operator <(const lib_descriptor& lhs, const lib_descriptor& rhs) {
	return lhs.fname_ < rhs.fname_;
}

// equality operator for lib's handlers
bool operator ==(const lib_descriptor& left, const lib_descriptor& right) {
	return left.fname_ == right.fname_;
}
	
}} /* namespace blue_sky::detail */

