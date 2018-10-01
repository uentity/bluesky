/// @file
/// @author uentity
/// @date 30.08.2016
/// @brief BlueSky log subsystem test
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include <bs/error.h>
#include <bs/log.h>

#include <spdlog/fmt/ostr.h>
#include <boost/test/unit_test.hpp>
#include <iostream>

struct example {
	friend std::ostream& operator <<(std::ostream& os, const example& ex) {
		return os << "hello";
	}
};

BOOST_AUTO_TEST_CASE(test_bs_log) {
	using namespace blue_sky;
	using namespace blue_sky::log;

	std::cout << "\n\n*** testing log..." << std::endl;

	//bsout().logger().info("{}", example());

	auto bs_out = bs_log("out");
	bs_out << info("Hello World! My name is {}", "Alexander") << end;
	int m = 2;
	bs_out << I("My age is {} years and {} months") << 35 << m << end;
	bserr() << W("My age is {} years and {} months", 35, m) << end;
	BSERROR << E("My age is {} years and {} months") << 35 << m << end;
	bserr() << C("My age is {} years and {} months") << 35 << m << end;
	bsout() << T("My age is {} years and {} months") << 35 << m << end;
	BSOUT << D("My age is {} years and {} months") << 35 << m << end;
	BSOUT << O("My age is {} years and {} months") << 35 << m << end;

	auto&& tape = I("Hello {} {}") << std::string("World") << 35;
	bsout() << tape << end;

	bsout() << "Hello simple {}" << "World" << end;
	bsout() << I("My name is {}") << "Alexander" << "and I'm {} years old" << 35 << end;

	bsout() << error("Kernel::test: Kernel panic!").to_string() << bs_end;
}

