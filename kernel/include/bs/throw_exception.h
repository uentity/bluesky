/// @file
/// @author Sergey Miryanov, Alexader Gagarin
/// @date 14.03.2017
/// @brief throw exception from bs_bos_core methods
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <boost/current_function.hpp>
#include <boost/lexical_cast.hpp>
#include "exception.h"
#include <string>

namespace blue_sky {

#ifdef _DEBUG
#define bs_throw_exception(msg) \
	throw bs_exception(msg, std::string(__FILE__) + ':' + boost::lexical_cast<std::string>(__LINE__) + ": [" + BOOST_CURRENT_FUNCTION + ']');
#else
#define bs_throw_exception(msg) \
	throw bs_exception(msg, std::string(__FILE__) + ":" + boost::lexical_cast<std::string>(__LINE__), msg);
#endif

}

