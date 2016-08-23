/// @file
/// @author uentity
/// @date 23.08.2016
/// @brief Variadic arguments prcessing helpers
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <utility>
#include <type_traits>

namespace blue_sky {

/*-----------------------------------------------------------------
 * extract params of given type and position from variadic sequence
 *----------------------------------------------------------------*/
struct bs_args {
	// argument contains: value type, default value and count in arguments list of value type
	template< typename T, int Seq_num = 0 >
	struct a {
		using value_type = T;
		static constexpr int seq_num = Seq_num;
	};

	// generic template, assume that T is specialized argument - never instantiated
	template< typename T, typename... Args >
	struct get;

	// when we have zero args - use default value
	template< typename T >
	struct get< T > {
		static auto value(typename T::value_type arg) {
			return arg;
		}
	};

	// at lest one argument in list
	template< typename T, typename ArgT1, typename... Args >
	struct get< T, ArgT1, Args... > {

		// counter checker
		// on each recursion level param's seq_num is decremented
		// and(!) next argument is stripped from args tail
		template< typename TT, typename ArgT, typename Enable = void >
		struct counter_impl : get< a< typename TT::value_type, TT::seq_num - 1 >, Args... > {
			using base_t = get< a< typename TT::value_type, TT::seq_num - 1 >, Args... >;

			static auto value(
				typename TT::value_type tar_a, typename ArgT::value_type t,
				typename Args::value_type... args
			) {
				return base_t::value(tar_a, args...);
			}
		};

		// if we reached zero counter - stop recursion and return value found
		template< typename TT, typename ArgT >
		struct counter_impl< TT, ArgT, std::enable_if_t< TT::seq_num == 0 > > {
			static auto value(
				typename TT::value_type tar_a, typename ArgT::value_type t,
				typename Args::value_type... args
			) {
				return t;
			}
		};

		// recursion over arguments when type didn't match - strip ArgT
		template< typename TT, typename ArgT, typename Enable = void >
		struct impl : get< TT, Args... > {
			static auto value(
				typename TT::value_type tar_a, typename ArgT::value_type t,
				typename Args::value_type... args
			) {
				// erase ArgT from args
				return get< TT, Args... >::value(tar_a, args...);
			}
		};

		// when we found given type - check counter
		template< typename TT, typename ArgT >
		struct impl<
			TT, ArgT,
			std::enable_if_t<
				std::is_same<
					std::decay_t< typename TT::value_type >, std::decay_t< typename ArgT::value_type >
				>::value
			>
		> : counter_impl< TT, ArgT >
		{};

		static auto value(
			typename T::value_type tar_a, typename ArgT1::value_type t,
			typename Args::value_type... args
		) {
			return impl< T, ArgT1 >::value(tar_a, t, args...);
		}
	};

	// sugar for simpl arguments -- 1st in list of each type
	template< typename T, typename... Args >
	static auto get_value(T&& tar_def_val, Args&&... args) {
		return get< a< T >, a< Args >... >::value(
			std::forward< T >(tar_def_val), std::forward< Args >(args)...
		);
	}

	// extended version allows to specify type and position of argument in list
	// using bs_args::a structures
	template< typename T, typename... Args >
	static auto get_value_ext(typename T::value_type tar_def_val, typename Args::value_type... args) {
		return get< T, Args... >::value(tar_def_val, args...);
	}
};

/*-----------------------------------------------------------------
 * configure some entity in compile time
 *----------------------------------------------------------------*/
struct bs_configurable {
	// argument handle contains: value type, default value and count in arguments list of value type
	template< typename T, T def_value = T(), int Seq_num = 0 >
	struct a : std::integral_constant< T, def_value > {
		static constexpr int seq_num = Seq_num;
	};

	// generic template, assume that T is specialized argument - never instantiated
	template< typename T, typename... Args >
	struct get_value;

	// when we have zero args - use default value
	template< typename T >
	struct get_value< T > : std::integral_constant< typename T::value_type, T::value > {};

	// at lest one argument in list
	template< typename T, typename ArgT1, typename... Args >
	struct get_value< T, ArgT1, Args... > {
		// check counter
		// on each recursion level param's seq_num is decremented
		template< typename TT, typename ArgT, typename Enable = void >
		struct counter_impl : get_value< a< typename TT::value_type, TT::value, TT::seq_num - 1 >, Args... >
		{};

		// if we reached zero counter - stop recursion
		template< typename TT, typename ArgT >
		struct counter_impl< TT, ArgT, std::enable_if_t< TT::seq_num == 0 > >
			: std::integral_constant< typename TT::value_type, ArgT::value >
		{};

		// recursion over arguments when type didn't match - strip ArgT
		template< typename TT, typename ArgT, typename Enable = void >
		struct impl : get_value< T, Args... > {};
		//	: std::integral_constant< T::value_type, get_value< T, Args... >::value > {
		//	static constexpr bool match = get_first_value< T, Args... >::match;
		//};

		// when we found given type
		template< typename TT, typename ArgT >
		struct impl<
			TT, ArgT,
			std::enable_if_t<
				std::is_same<
					std::decay_t< typename TT::value_type >, std::decay_t< typename ArgT::value_type >
				>::value
			>
		> : counter_impl< TT, ArgT >
		{};

		static constexpr auto value = impl< T, ArgT1 >::value;
	};

	template< typename T, typename... Args >
	static auto value(T tar_def_val, Args... args) {
		return get_value< a< T, tar_def_val >, a< Args, args >... >::value;
	}

	// assume that T is param instance
	//template< typename T, typename... Args >
	//static constexpr auto value() {
	//	return get_value< T, Args... >::value;
	//}
};

} /* namespace blue_sky */
