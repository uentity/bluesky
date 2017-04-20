/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief BlueSky logging framework - based on spdlog library
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "common.h"
#include "detail/apply.h"
#include <spdlog/logger.h>

namespace blue_sky { namespace log {

// import spdlog level_enum
using level_enum = spdlog::level::level_enum;
template< level_enum level > using level_const = std::integral_constant< level_enum, level>;

class BS_API bs_log {
public:
	using manip_t = bs_log& (*)(bs_log&);

	template< level_enum Level, typename... Args >
	struct log_tape {
		// helper to find out type of tape without first param (bs_log&)
		template< typename T, typename... Ts >
		struct rem_tape_head;

		template< typename T >
		struct rem_tape_head< T > {};

		template< typename T, typename... Ts >
		struct rem_tape_head {
			//using tuple_t = std::tuple< Ts... >;
			using type = log_tape< Level, Ts... >;
			using write_overl_t = decltype(&type::template write< Ts... >);
			static constexpr write_overl_t write_overl = &type::template write< Ts... >;
		};

		// construct from args tuple
		log_tape(std::tuple< Args... > args_tup) :
			tape_(std::move(args_tup))
		{}

		// append next value to tape
		// enabled for all types except manipulators
		template<
			typename T,
			typename = std::enable_if_t< !std::is_same< std::decay_t<T>, manip_t >::value >
		>
		decltype(auto) operator <<(T&& data) const {
			return log_tape< Level, Args..., T >(
				std::tuple_cat(tape_, std::forward_as_tuple(data))
			);
		}

		// overload for manipulators
		bs_log& operator <<(manip_t op) const {
			// flush data and call manipulator
			return (*op)(flush());
		}

		// flush self to backend spdlog::logger
		// assume that bs_log& is the first tape entry
		bs_log& flush() const {
			//using params_tape_t = typename rem_tape_head< Args... >::type;
			bs_log& L = std::get<0>(tape_);
			apply(
				rem_tape_head< Args... >::write_overl,
				std::tuple_cat(
					std::forward_as_tuple(L.log_),
					// cut 1st tuple argument -- bs_log reference
					subtuple(tape_, std::integral_constant< std::size_t, 1 >())
				)
			);
			return L;
		}

		// write (forward) raw arguments to spdlog::logger backend
		template< typename... Ts >
		static void write(spdlog::logger& L, Ts&&... args) {
			L.log(Level, std::forward< Ts >(args)...);
		}

		std::tuple< Args... > tape_;
	};

	// overload << for manipulators
	BS_API friend bs_log& operator<<(bs_log& lhs, manip_t op) {
		return (*op)(lhs);
	}

	// overload for log_tape
	template< level_enum Level, typename... Args >
	friend decltype(auto)
	operator <<(bs_log& lhs, const log_tape< Level, Args... >& rhs) {
		return log_tape< Level, bs_log&, Args... >(
			std::tuple_cat(std::forward_as_tuple(lhs), rhs.tape_)
		);
	}

	// overload for arbitrary type just costructs log_tape
	// with 'info' level
	template< class T >
	friend decltype(auto) operator <<(bs_log& lhs, const T& rhs) {
		return log_tape< level_enum::info, bs_log&, const T& >(
			std::tuple_cat(std::forward_as_tuple(lhs), std::forward_as_tuple(rhs))
		);
	}

	bs_log(spdlog::logger& log) : log_(log) {}
	bs_log(const char* name);

	/// @brief Create spdlog::logger backend with given name
	///
	/// @param name
	static spdlog::logger& get_logger(const char* name);

	// access spdlog::logger backend
	spdlog::logger& logger() {
		return log_;
	}
	const spdlog::logger& loggger() const {
		return log_;
	}

	// return log level
	level_enum level() const {
		return log_.level();
	}

private:
	spdlog::logger& log_;
};

/*-----------------------------------------------------------------
 * Manipulators
 *----------------------------------------------------------------*/
// end manipulator does nothing
BS_API bs_log& end(bs_log& l);

// log level manipulators
BS_API bs_log& infol(bs_log& l);
BS_API bs_log& warnl(bs_log& l);
BS_API bs_log& errl(bs_log& l);
BS_API bs_log& criticall(bs_log& l);
BS_API bs_log& offl(bs_log& l);
BS_API bs_log& tracel(bs_log& l);
BS_API bs_log& debugl(bs_log& l);

/*-----------------------------------------------------------------
 * Log levels
 *----------------------------------------------------------------*/
namespace detail {

// generic log_tape constructor for any given level
template< level_enum level, typename... Args >
decltype(auto) make_log_tape(level_const< level >, Args&&... args) {
	return bs_log::log_tape< level, Args... >(std::forward_as_tuple(args...));
}

} // eof namespace detail

// info
template< typename... Args >
decltype(auto) info(Args&&... args) {
	return detail::make_log_tape(
		level_const< level_enum::info >(), std::forward< Args >(args)...
	);
}
template< typename... Args >
decltype(auto) I(Args&&... args) {
	return info(std::forward< Args >(args)...);
}

// warn
template< typename... Args >
decltype(auto) warn(Args&&... args) {
	return detail::make_log_tape(
		level_const< level_enum::warn >(), std::forward< Args >(args)...
	);
}
template< typename... Args >
decltype(auto) W(Args&&... args) {
	return warn(std::forward< Args >(args)...);
}

// error
template< typename... Args >
decltype(auto) err(Args&&... args) {
	return detail::make_log_tape(
		level_const< level_enum::err >(), std::forward< Args >(args)...
	);
}
template< typename... Args >
decltype(auto) E(Args&&... args) {
	return err(std::forward< Args >(args)...);
}

// critial
template< typename... Args >
decltype(auto) critical(Args&&... args) {
	return detail::make_log_tape(
		level_const< level_enum::critical >(), std::forward< Args >(args)...
	);
}
template< typename... Args >
decltype(auto) C(Args&&... args) {
	return critical(std::forward< Args >(args)...);
}

// trace
template< typename... Args >
decltype(auto) trace(Args&&... args) {
	return detail::make_log_tape(
		level_const< level_enum::trace >(), std::forward< Args >(args)...
	);
}
template< typename... Args >
decltype(auto) T(Args&&... args) {
	return trace(std::forward< Args >(args)...);
}

// debug
template< typename... Args >
decltype(auto) debug(Args&&... args) {
	return detail::make_log_tape(
		level_const< level_enum::debug >(), std::forward< Args >(args)...
	);
}
template< typename... Args >
decltype(auto) D(Args&&... args) {
	return debug(std::forward< Args >(args)...);
}

// off
template< typename... Args >
decltype(auto) off(Args&&... args) {
	return detail::make_log_tape(
		level_const< level_enum::off >(), std::forward< Args >(args)...
	);
}
template< typename... Args >
decltype(auto) O(Args&&... args) {
	return off(std::forward< Args >(args)...);
}

} // eof blue_sky::log namespace

/*-----------------------------------------------------------------
 * Functions for quick access of main out & error logs
 *----------------------------------------------------------------*/
BS_API log::bs_log& bsout();
BS_API log::bs_log& bserr();

#define BSOUT ::blue_sky::bsout()
#define BSERROR ::blue_sky::bserr()

// import log::end as bs_end
#define bs_end ::blue_sky::log::end

} /* namespace blue_sky */

