/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Helper macro for use in BS plugins
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "setup_common_api.h"

#ifdef BS_API_PLUGIN 
	#undef BS_API_PLUGIN
#endif
#ifdef BS_C_API_PLUGIN 
	#undef BS_C_API_PLUGIN
#endif
#ifdef BS_HIDDEN_API_PLUGIN 
	#undef BS_HIDDEN_API_PLUGIN
#endif

//setup plugins API macro
#ifdef BS_EXPORTING_PLUGIN
	#define BS_API_PLUGIN _BS_API_EXPORT
	#define BS_C_API_PLUGIN _BS_C_API_EXPORT
	#define BS_HIDDEN_API_PLUGIN _BS_HIDDEN_API_EXPORT
#else
	#define BS_API_PLUGIN _BS_API_IMPORT
	#define BS_C_API_PLUGIN _BS_C_API_IMPORT
	#define BS_HIDDEN_API_PLUGIN _BS_HIDDEN_API_IMPORT
#endif

