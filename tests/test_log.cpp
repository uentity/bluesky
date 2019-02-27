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

#include <fmt/format.h>
#include <fmt/locale.h>
#include <fmt/ostream.h>
#include <boost/test/unit_test.hpp>
#include <iostream>

struct log_example {
	static std::size_t ctor_count, copy_count, move_count;

	log_example() {
		++ctor_count;
	}

	log_example(const log_example&) {
		++copy_count;
	}

	log_example(log_example&&) {
		++move_count;
	}

	friend std::ostream& operator <<(std::ostream& os, const log_example& ex) {
		return os << "log_example: ctor " << ctor_count << ", copy " << copy_count << ", move " << move_count;
	}
};

std::size_t log_example::ctor_count = 0;
std::size_t log_example::move_count = 0;
std::size_t log_example::copy_count = 0;

BOOST_AUTO_TEST_CASE(test_bs_log) {
	using namespace blue_sky;
	using namespace blue_sky::log;

	std::cout << "\n\n*** testing log..." << std::endl;
	bsout() << "*** Locale: {}" << std::locale("").name() << bs_end;
	bsout() << "BS log: {} | classic locale: {} | cur locale: {}" << 3.14
		<< fmt::format(std::locale::classic(), "{}", 3.14)
		<< fmt::format(std::locale(""), "{}", 3.14)
		<< bs_end;
	bsout() << "INSERT OR REPLACE INTO wells (name, x, y, KB, src) VALUES('{}', {}, {}, {}, {})" <<
		154405 << 154405 << 68.71 << 0 << "1G" << bs_end;

	// test logger ability to print arbitrary type
	auto e = log_example();
	bsout().logger().info("{}", e);

	auto bs_out = bs_log("out");
	int y = 35, m = 2;
	std::string kitty = "kitty";
	const char* city = "city";

	// construct big tape at once
	auto phrase = "It's a pity that a little {} lives in a big {} for {} years and {} months";
	bs_out << info(phrase, "kitty", "city", y, 2) << end;
	// test dieeferent log levels and log streams
	bs_out  << D(phrase, "kitty", "city") << 35 << m << end;
	bserr() << W(phrase, "kitty", "city") << y << m << end;
	bserr() << C(phrase) << "kitty" << "city" << y << m << end;
	bsout() << T(phrase) << kitty << city << y << m << end;
	BSOUT   << O(phrase) << std::string("kitty") << city << y << m << end;
	BSERROR << E(phrase) << std::string("kitty") << std::string(city) << y << m << end;

	// test tape working with elements
	auto tape1 = W("Exepect {} ctors, {} copies, {} moves | {}") << 2 << 1 << 4 << log_example();
	bsout().logger().info("{}", e);
	bs_out << tape1 << end;
	// tape concatenation -- isn't work as expected and does not print tape1 at all.
	auto tape2 = I(phrase) << kitty << city << 35 << 2 << tape1;
	bs_out << std::move(tape2) << end;

	//// nothig should change
	//bs_out << tape1 << end;
	//// make longer tape
	//auto&& tape2 = tape1 << ", one more " << log_example();
	//bs_out << tape2 << end;

	auto&& tape = I("Hello {} {}") << std::string("World") << 35;
	bsout() << tape << end;

	bsout() << "Hello simple {}" << "World" << end;
	// next sentense won't print what expected
	bsout() << I("It's a pity that a little {} {}") << "kitty"
		<< "lives in a big {} for {}.{} years" << "city" << y << m << end;

	bsout() << error("Kernel::test: Kernel panic!").to_string() << bs_end;
}

