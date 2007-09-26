// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#include "bs_common.h"
#include "bs_report.h"
#include "bs_kernel.h"
#include <boost/python/module_init.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <string>
#include <exception>

#ifdef _WIN32
	#include <windows.h>
	#include <Psapi.h>
#else
	#include <dlfcn.h>
#endif

//! define TARGET_NAME in compiler equal to output loader's name without extension
#ifndef TARGET_NAME
#define TARGET_NAME bs
#endif

#define S_TARGET_NAME BOOST_PP_STRINGIZE(TARGET_NAME)
#define INIT_FN_NAME BOOST_PP_CAT(init, TARGET_NAME)

using namespace blue_sky;
using namespace std;

template< class sym_t >
int load_sym_glob(const char* sym_name, sym_t& sym) {
#ifdef UNIX
	sym = (sym_t)dlsym(RTLD_DEFAULT, sym_name);
	return 0;
#elif defined(_WIN32)
	//helper struct to find BlueSky kernel among all loaded modules
	struct find_kernel_module {
		static HMODULE go() {
			HANDLE hproc;
			HMODULE hmods[1024];
			DWORD cbNeeded;
			BS_GET_PLUGIN_DESCRIPTOR pdf = NULL;
			plugin_descriptor* pd = NULL;
			ulong m_ind = 0;

			//get handle of current process
			hproc = GetCurrentProcess();

			//enumerate all modules of current process
			if(hproc && EnumProcessModules(hproc, hmods, sizeof(hmods), &cbNeeded))	{
				ulong cnt = cbNeeded / sizeof(HMODULE);
				for (ulong i = 0; i < cnt; ++i) {
					//search for given symbol in i-th module
					if(pdf = (BS_GET_PLUGIN_DESCRIPTOR)GetProcAddress(hmods[i], "bs_get_plugin_descriptor")) {
						//get pointer to plugin_descriptor & check if this is a kernel
						if((pd = pdf()) && pd->name_.compare("BlueSky kernel") == 0) 
							return hmods[i];
					}
				}
				CloseHandle(hproc); 
			}
			return NULL;
		}
	};

	//get kernel module handle
	static HMODULE km = find_kernel_module::go();

	sym = NULL;
	if(!km) return 1;

	//search for given symbol
	sym = (sym_t)GetProcAddress(km, sym_name);
	return 0;
#endif
}

BS_C_API_PLUGIN void INIT_FN_NAME() {
	//search for BlueSky's kernel plugin descriptor
	BS_GET_PLUGIN_DESCRIPTOR get_pd_fn;
	if(load_sym_glob("bs_get_plugin_descriptor", get_pd_fn) != 0 || !get_pd_fn) {
		BSERROR << "BlueSky kernel descriptor wasn't found or invalid!" << bs_end;
		return;
		//throw std::runtime_error("BlueSky kernel descriptor wasn't found");
	}
	plugin_descriptor* kernel_pd = get_pd_fn();
	//correct global namespace to match TARGET_NAME (otherwise Python throws an error)
	kernel_pd->py_namespace_ = S_TARGET_NAME;

	//load plugins with Python subsystem
	give_kernel::Instance().LoadPlugins(true);
}
