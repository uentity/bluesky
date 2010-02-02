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

#ifdef _WIN32
#include "windows.h"
#include <locale.h>
#endif

#include "bs_common.h"
#include "bs_kernel.h"
#include "bs_report.h"

using namespace blue_sky;

//kernel* g_kernel;
//bs_log& g_log = log::Instance();
//kernel& g_kernel = give_kernel::Instance();

#ifndef UNIX
BOOL WINAPI DllMain(HANDLE hInst, DWORD rReason, LPVOID ipReserved)
{
	//kernel& k = give_kernel::Instance();
  switch(rReason)
    {
    case DLL_PROCESS_ATTACH:
      setlocale(LC_ALL,"Russian");
      break;
    case DLL_PROCESS_DETACH:
      break;
    default:break;
    }

  return true;
}

#else
//before main-function execution
__attribute__ ((constructor)) void bs_init()
{
}

//after main-function execution
__attribute__ ((destructor)) void bs_fini()
{
}

#endif

