/// @file
/// @author uentity
/// @date 23.08.2016
/// @brief Common unit test unit
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include <bs/error.h>
#include <bs/detail/args.h>

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace blue_sky;
using namespace boost::unit_test;

template< typename T, int Seq_num = 0 >
using a = bs_args::a< T, Seq_num >;

template< typename T, int Count = 0, typename... Args >
struct args_printer {
	template< int Seq_num, typename Enable = void >
	struct impl {
		static void go(T def_value, Args... args) {
			std::cout << bs_args::get_value_ext< a< T, Seq_num >, a< Args >... >(def_value, args...) << ' ';
			// recursion
			impl< Seq_num + 1 >::go(def_value, args...);
		}
	};

	template< int Seq_num >
	struct impl< Seq_num, std::enable_if_t< Seq_num >= Count > > {
		static void go(T def_value, Args... args) {
			// stop recursion
			std::cout << std::endl;
		}
	};

	static void go(T def_value, Args... args) {
		impl< 0 >::go(def_value, args...);
	}
};

template< typename T, int Count = 0, typename... Args >
void print_args(T def_value, Args... args) {
	args_printer< T, Count, Args... >::go(def_value, args...);
}

template< typename... Args >
void parse_args(Args... args) {
	std::cout << "integers: ";
	print_args< int, 4, Args... >(-888, args...);
	std::cout << "const char*: ";
	print_args< const char*, 3, Args... >("error", args...);
	std::cout << "char*: ";
	const char* def_str = "error";
	print_args< char*, 1, Args... >(const_cast< char* >(def_str), args...);
	std::cout << "std::string: ";
	print_args< std::string, 2, Args...>("error", args...);
	std::cout << "float: ";
	print_args< float, 2, Args...>(-888.f, args...);


	// simple test - extract first args in list
	BOOST_TEST(bs_args::get_value(-888, args...) == -42);
	BOOST_TEST(bs_args::get_value("error", args...) == "hello");
	BOOST_TEST(bs_args::get_value(-888.f, args...) == 42.f);
	BOOST_TEST(bs_args::get_value(std::string("error"), args...) == "std::string");

	// extract more ints
	auto r1 = bs_args::get_value_ext< a< int, 1 >, a< Args >... >(-888, args...);
	BOOST_TEST(r1 == 0);
	auto r2 = bs_args::get_value_ext< a< int, 2 >, a< Args >... >(-888, args...);
	BOOST_TEST(r2 == 2);
	// extract int ptr
	int error = -888;
	auto r3 = bs_args::get_value_ext< a< int* >, a< Args >... >(&error, args...);
	BOOST_TEST(*r3 == 1);

	// extract 2nd std::string
	auto r4 = bs_args::get_value_ext< a< std::string, 1 >, a< Args >... >("error", args...);
	BOOST_TEST(r4 == "!");

	// extract 2nd fp value
	auto r6 = bs_args::get_value_ext< a< float, 1 >, a< Args >... >(-888.f, args...);
	BOOST_TEST(r6 == 43.f);
	// we have float arguments, not double
	auto r5 = bs_args::get_value_ext< a< double, 1 >, a< Args >... >(-888.f, args...);
	BOOST_TEST(r5 == -888.f);
}

BOOST_AUTO_TEST_CASE(test_bs_args) {
	std::cout << "\n\n*** testing bs_args..." << std::endl;

	int my_ints[] = {0, 1, 2};
	std::string my_stdstr = "std::string";
	const char* my_str = "string";
	char* my_modstr = const_cast< char* >(my_str);

	parse_args(
		-42, "hello", my_ints[0], "world", &my_ints[1], 42.f, 43.f,
		static_cast< const int& >(my_ints[2]), my_str, my_modstr, 42,
		my_stdstr, std::string("!")
	);
}

