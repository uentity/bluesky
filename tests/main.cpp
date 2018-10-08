/// @file
/// @author uentity
/// @date 23.08.2016
/// @brief Main unit of test module
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE blue_sky unit test
//#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <bs/kernel.h>
#include <bs/plugin_descriptor.h>
#include <boost/test/unit_test.hpp>

BS_PLUGIN_DESCRIPTOR("unit_test", "1.0", "Unit-test plugin");

BOOST_AUTO_TEST_CASE(test_main) {
	BS_KERNEL.init();
	BOOST_CHECK(BS_KERNEL.register_plugin(bs_get_plugin_descriptor()));
}

