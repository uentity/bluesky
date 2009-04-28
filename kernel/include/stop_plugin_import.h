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

#include "setup_common_api.h"
#include "boost/preprocessor/slot/slot.hpp"

#if defined(_BS_PL_IMPORTS_CNT)
	#if _BS_PL_IMPORTS_CNT == 1
		#undef _BS_PL_IMPORTS_CNT
		#ifndef BS_EXPORTING_PLUGIN
			#define BS_EXPORTING_PLUGIN
			#include BS_SETUP_PLUGIN_API()
		#endif
	#else
		//_BS_PL_IMPORTS_CNT = (_BS_PL_IMPORTS_CNT - 1)
		#define BOOST_PP_VALUE _BS_PL_IMPORTS_CNT - 1
		#include BOOST_PP_ASSIGN_SLOT(1)
		#undef _BS_PL_IMPORTS_CNT
		#define _BS_PL_IMPORTS_CNT BOOST_PP_SLOT(1)
	#endif
#endif
