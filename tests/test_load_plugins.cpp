/// @file
/// @author uentity
/// @date 01.05.2017
/// @brief Plugis loading machinery check
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include <bs/exception.h>
#include <bs/kernel.h>

#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE(test_load_plugins) {
	BS_KERNEL.load_plugins();
}

