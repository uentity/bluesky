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

#include <bs/log.h>
#include <bs/kernel/misc.h>
#include <bs/kernel/plugins.h>
#include <bs/plugin_descriptor.h>
#include <boost/test/unit_test.hpp>

#include "test_objects.h"

BS_PLUGIN_DESCRIPTOR("unit_test", "1.0", "Unit-test plugin");

// forward declare fixture types registering function
namespace blue_sky { void register_test_objects(); }

BS_REGISTER_PLUGIN {
	using namespace blue_sky;
	BSOUT << "*** BS tests global initialization" << bs_end;

	std::setlocale(LC_ALL, "ru_RU.UTF-8");

	// init kernel & register test plugins
	kernel::init();
	BOOST_CHECK(kernel::plugins::register_plugin(bs_get_plugin_descriptor()));

	// explicitly init serialization subsystem
	// [UPDATE] not needed, because auto-invoked by `register_plugin()`
	//kernel::unify_serialization();

	// register fixture objects constructors, etc
	register_test_objects();

	return true;
}

struct bs_global_fixture {
	void setup() { bs_register_plugin( {bs_get_plugin_descriptor()} ); }

	void teardown() {}
};

BOOST_TEST_GLOBAL_FIXTURE(bs_global_fixture);

