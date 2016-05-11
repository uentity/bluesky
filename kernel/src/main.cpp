/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/common.h>
//#include <bs/kernel.h>
//#include <bs/report.h>

//kernel* g_kernel;
//bs_log& g_log = log::Instance();
//kernel& g_kernel = give_kernel::Instance();

#ifdef _WIN32
#include "windows.h"
#include <locale.h>

BOOL WINAPI DllMain(HANDLE hInst, DWORD rReason, LPVOID ipReserved) {
	//kernel& k = give_kernel::Instance();
	switch(rReason) {
		case DLL_PROCESS_ATTACH:
			setlocale(LC_ALL,"Russian");
			break;
		case DLL_PROCESS_DETACH:
		default:
			break;
	}
	return true;
}

#else
//before main-function execution
__attribute__ ((constructor)) void bs_init()
{}

//after main-function execution
__attribute__ ((destructor)) void bs_fini()
{}

#endif

