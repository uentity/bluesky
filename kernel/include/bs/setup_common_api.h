/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Define BlueSky kernel & plugins library linkage macro
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

// setup API macro depending on compiler
#if defined(_MSC_VER)		//M$ compiler
	// disable warning about inheriting from non-DLL types - can be ignored for STL (apply to bs_exception)
	#pragma warning(disable:4275)
	// disable xxx needs to have dll-interface blah-blah
	#pragma warning(disable:4251)
	// deprecation is not an error (for pybind11)
	#pragma warning(disable:4996)
	// 4003 is triggered by Boost.Preprocessor
	#pragma warning(disable:4003)

	// API export/import macro
	#define _BS_API_EXPORT __declspec(dllexport)
	#define _BS_C_API_EXPORT extern "C" __declspec(dllexport)
	#define _BS_HIDDEN_API_EXPORT

	#define _BS_API_IMPORT __declspec(dllimport)
	#define _BS_C_API_IMPORT extern "C" __declspec(dllimport)
	#define _BS_HIDDEN_API_IMPORT

#elif defined(__GNUC__)		//GCC
	// API export/import macro

	#define _BS_API_EXPORT __attribute__ ((visibility("default")))
	#define _BS_C_API_EXPORT extern "C" __attribute__ ((visibility("default")))
	#define _BS_HIDDEN_API_EXPORT __attribute__ ((visibility("hidden")))

	#define _BS_API_IMPORT
	#define _BS_C_API_IMPORT
	#define _BS_HIDDEN_API_IMPORT
#endif

// setup kernel API macro
#ifdef BS_EXPORTING
	#define BS_API _BS_API_EXPORT
	#define BS_C_API _BS_C_API_EXPORT
	#define BS_HIDDEN_API _BS_HIDDEN_API_EXPORT
#else
	#define BS_API _BS_API_IMPORT
	#define BS_C_API _BS_C_API_IMPORT
	#define BS_HIDDEN_API _BS_HIDDEN_API_IMPORT
#endif

// macro for plugin's API setup
#define BS_SETUP_PLUGIN_API() <bs/setup_plugin_api.h>
#define BS_START_PLUGIN_IMPORT() <bs/force_plugin_import.h>
#define BS_STOP_PLUGIN_IMPORT() <bs/stop_plugin_import.h>

