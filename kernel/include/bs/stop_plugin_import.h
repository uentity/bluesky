/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "setup_common_api.h"
#include <boost/preprocessor/slot/slot.hpp>

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
