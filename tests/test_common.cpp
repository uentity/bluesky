/// @file
/// @author uentity
/// @date 23.08.2016
/// @brief Common unit test unit
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include "test_serialization.h"

#include <bs/error.h>
#include <bs/detail/args.h>
#include <bs/timetypes.h>
#include <bs/log.h>
#include <bs/detail/function_view.h>

#include <boost/test/unit_test.hpp>
#include <fmt/ostream.h>
#include <fmt/format.h>
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

int adder_fn(int x1, int x2) {
	return x1 + x2;
}

auto invoke_function_view(function_view<int (int, int)> f, int x1, int x2) {
	return f(x1, x2);
}

BOOST_AUTO_TEST_CASE(test_common) {
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

	// test time types
	auto ts = make_timestamp();
	BOOST_TEST(to_string(ts - ts) == "0ns");
	std::cout << "Direct print timestamp: " << to_string(ts) << std::endl;
	//bsout() << "Print timestamp using fmt: {}" << ts << bs_end;

	// test function_view
	auto fv1 = function_view{adder_fn};
	BOOST_TEST(fv1(42, 42) == 84);
	BOOST_TEST(invoke_function_view(fv1, 42, 42) == 84);
	BOOST_TEST(invoke_function_view(adder_fn, 42, 42) == 84);

	auto f2 = [x0 = 0](int x1, int x2) { return x0 + x1 + x2; };
	auto fv2 = function_view{f2};
	BOOST_TEST(fv2(42, 42) == 84);
	BOOST_TEST(invoke_function_view(fv2, 42, 42) == 84);
	BOOST_TEST(invoke_function_view(f2, 42, 42) == 84);

	auto f3 = std::function<int (int, int)>(adder_fn);
	auto fv3 = function_view{f3};
	BOOST_TEST(fv3(42, 42) == 84);
	BOOST_TEST(invoke_function_view(fv3, 42, 42) == 84);
	BOOST_TEST(invoke_function_view(f3, 42, 42) == 84);

	auto f4 = std::plus<int>();
	auto fv4 = function_view{f4};
	BOOST_TEST(fv4(42, 42) == 84);
	BOOST_TEST(invoke_function_view(fv4, 42, 42) == 84);
	BOOST_TEST(invoke_function_view(f4, 42, 42) == 84);

	// compile failure
	//auto f4 = function_view{42.};

	// test error
	auto er = error{"Something bad", -1};
	auto ec = make_error_code(Error::Happened);
	BOOST_TEST(er.code.value() == -1);
	BOOST_TEST(er.message() == "Something bad");
	BOOST_TEST(er.domain() == ec.category().name());
	BOOST_TEST(er.what() == "|> Something bad");
	// test serialization
	auto er1 = success(test_json(er.pack()));
	BOOST_TEST(er.code == er1.code);
	BOOST_TEST(er.domain() == er1.domain());
	BOOST_TEST(er.what() == er1.what());

	struct like_bool {
		operator bool() const { return true; }
	};

	struct like_var : std::variant<double, error> {
		using variant::variant;
	};

	static const auto ff = []() { return like_bool{}; };
	auto er2 = error(static_cast<bool>(like_bool{}));
	auto er3 = error(like_var(perfect));
	auto er31 = error(std::variant<error::box, int>{success().pack()});
	auto er4 = error(result_or_err<int>{unexpected_err_quiet(perfect)});
	auto er41 = error(result_or_errbox<int>{unexpected_err_quiet(perfect)});
	error::eval(ff);
}

