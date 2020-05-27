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
#include "detail/tuple_utils.h"

#include <spdlog/logger.h>

namespace blue_sky { namespace log {

/// Create spdlog::logger backend with given name
BS_API auto get_logger(const char* name) -> std::shared_ptr<spdlog::logger>;

// import spdlog level_enum
using level_enum = spdlog::level::level_enum;
template< level_enum level > using level_const = std::integral_constant< level_enum, level>;

// forward declarations
// main log class
class bs_log;
// manipulator fucntion type
using manip_t = bs_log& (*)(bs_log&);

NAMESPACE_BEGIN(detail)
/*-----------------------------------------------------------------------------
 *  helper that collects all arguments to be logged into a tape (tuple)
 *  prefixed by `bs_log` reference
 *-----------------------------------------------------------------------------*/
template<level_enum Level, typename... Args> struct log_tape;

///////////////////////////////////////////////////////////////////////////////
// check if type is log tape
//
template<typename T>
struct is_log_tape_impl : std::false_type {};

template<level_enum Level, typename... Args>
struct is_log_tape_impl<log_tape<Level, Args...>> : std::true_type {};

template<typename T>
constexpr auto is_log_tape() -> bool
{ return is_log_tape_impl<std::decay_t<T>>::value; }

///////////////////////////////////////////////////////////////////////////////
// extract tape tuple if `log_tape` is passed, otherwise do arg passthrough
//
template<typename T>
constexpr decltype(auto) make_tape_elem(
	T&& t,
	std::enable_if_t<is_log_tape<T>()>* = nullptr
) {
	return std::forward<T>(t).tape_;
}

template<typename T>
constexpr decltype(auto) make_tape_elem(
	T&& t,
	std::enable_if_t<!is_log_tape<T>(), int>* = nullptr
) {
	return std::forward<T>(t);
}

///////////////////////////////////////////////////////////////////////////////
//  `log_tape` implementation
//
template< level_enum Level, typename... Args >
struct log_tape {
private:
	template<typename A1 = void, typename... As>
	static constexpr bool head_is_log =
		std::is_same_v<std::remove_reference_t<A1>, bs_log>;

public:
	using tape_tuple_t = std::tuple<Args...>;

	// perfect forwarding ctor for underlying tuple
	template<typename... Ts>
	constexpr log_tape(Ts&&... args) :
		tape_(std::forward<Ts>(args)...)
	{}

	// generate tapes of this Level inferring tape elements from passed tuple
	template<typename... Ts>
	constexpr static auto create(std::tuple<Ts...>&& t) {
		return log_tape<Level, Ts...>(std::move(t));
	}

	// insert value in the beginning of tape
	// [NOTE] rvalues are copied into tape, lvalues are stored as references
	template<typename T>
	constexpr auto prepend(T&& data) const & {
		return create(grow_tuple(
			make_tape_elem(std::forward<T>(data)), tape_
		));
	}
	// overload for rvalue tapes, allows moving from this instead of copying
	template<typename T>
	constexpr auto prepend(T&& data) && {
		return create(grow_tuple(
			make_tape_elem(std::forward<T>(data)), std::move(tape_)
		));
	}

	// append next value to tape tail
	// [NOTE] rvalues are copied into tape, lvalues are stored as references
	template<typename T>
	constexpr auto append(T&& data) const & {
		return create(grow_tuple(
			tape_, make_tape_elem(std::forward<T>(data))
		));
	}
	// overload for rvalue tapes, allows moving from this instead of copying
	template<typename T>
	constexpr auto append(T&& data) && {
		return create(grow_tuple(
			std::move(tape_), make_tape_elem(std::forward<T>(data))
		));
	}

	// append next value to tape tail
	// enabled for all non-tuple types except manipulators
	// [NOTE] rvalues are copied into tape, lvalues are stored as references
	template<
		typename T,
		typename = std::enable_if_t< !std::is_same< std::decay_t<T>, manip_t >::value >
	>
	constexpr auto operator <<(T&& data) const &{
		return append(std::forward<T>(data));
	}
	// overload for rvalue tapes, allows moving from this instead of copying
	template<
		typename T,
		typename = std::enable_if_t< !std::is_same< std::decay_t<T>, manip_t >::value >
	>
	constexpr auto operator <<(T&& data) && {
		return append(std::forward<T>(data));
	}

	// overload for manipulators
	constexpr decltype(auto) operator <<(manip_t op) const {
		if constexpr(head_is_log<Args...>) {
			// flush data and call manipulator
			return (*op)(flush());
		}
		else (void)op;
	}

	// flush self to backend spdlog::logger
	constexpr auto flush() const -> bs_log& {
		return apply(write<Args...>, tape_);
	}

	// write (forward) raw arguments to spdlog::logger backend
	// [NOTE] taking tape elemants by const lvalue refs for printing
	template< typename BsLog, typename... Ts >
	constexpr static auto write(BsLog& L, const Ts&... args) -> BsLog& {
		L.logger().log(Level, args...);
		return L;
	}

	// tuple that collects data to log
	const tape_tuple_t tape_;
};

///////////////////////////////////////////////////////////////////////////////
//  log_tape generators from given set of arguments
//

// construct tape from arbitrary set of elements
template< level_enum level, typename... Args >
constexpr auto make_log_tape(Args&&... args) {
	return log_tape< level, Args... >(std::forward<Args>(args)...);
}

// construct tape with `Info` level from generic non-tape arguments
template<typename L, typename R>
constexpr auto make_log_tape(
	L&& lhs, R&& rhs,
	std::enable_if_t<!is_log_tape<L>() && !is_log_tape<R>()>* = nullptr
) {
	return log_tape< level_enum::info, L, R >(
		std::forward<L>(lhs), std::forward<R>(rhs)
	);
}

// append argument to lhs tape
template<typename L, typename R>
constexpr auto make_log_tape(
	L&& lhs, R&& rhs,
	std::enable_if_t<is_log_tape<L>(), short>* = nullptr
) {
	return std::forward<L>(lhs).append(std::forward<R>(rhs));
}

// prepend rhs tape with lhs argument
template<typename L, typename R>
constexpr auto make_log_tape(
	L&& lhs, R&& rhs,
	std::enable_if_t<!is_log_tape<L>() && is_log_tape<R>(), int>* = nullptr
) {
	return std::forward<R>(rhs).prepend(std::forward<L>(lhs));
}

NAMESPACE_END(detail)

/*-----------------------------------------------------------------------------
 *  `bs_log` wraps `spdlog::logger` with stream-like API
 *-----------------------------------------------------------------------------*/
class BS_API bs_log {
public:
	/// import detail::log_tape as tape
	template<level_enum Level, typename... Args> using tape = detail::log_tape<Level, Args...>;

	bs_log(std::shared_ptr<spdlog::logger> L);
	bs_log(const char* name);

	/// returns a log tape with reference to this as first element
	template<
		typename T,
		typename = std::enable_if_t<!std::is_same< std::decay_t<T>, manip_t >::value>
	>
	constexpr friend auto
	operator <<(bs_log& lhs, T&& rhs) {
		return detail::make_log_tape(lhs, std::forward<T>(rhs));
	}

	/// overload << for manipulators
	constexpr friend auto
	operator<<(bs_log& lhs, manip_t op) -> bs_log& {
		return (*op)(lhs);
	}

	// access spdlog::logger backend
	spdlog::logger& logger() {
		return *log_;
	}
	const spdlog::logger& loggger() const {
		return *log_;
	}

	// return log level
	level_enum level() const {
		return log_->level();
	}

private:
	// [NOTE] non-null invariant is enforced by constructors
	std::shared_ptr<spdlog::logger> log_;
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
// info
template< typename... Args >
constexpr auto info(Args&&... args) {
	return detail::make_log_tape<level_enum::info>(
		std::forward< Args >(args)...
	);
}
template< typename... Args >
constexpr auto I(Args&&... args) {
	return info(std::forward< Args >(args)...);
}

// warn
template< typename... Args >
constexpr auto warn(Args&&... args) {
	return detail::make_log_tape<level_enum::warn>(
		std::forward< Args >(args)...
	);
}
template< typename... Args >
constexpr auto W(Args&&... args) {
	return warn(std::forward< Args >(args)...);
}

// error
template< typename... Args >
constexpr auto err(Args&&... args) {
	return detail::make_log_tape<level_enum::err>(
		std::forward< Args >(args)...
	);
}
template< typename... Args >
constexpr auto E(Args&&... args) {
	return err(std::forward< Args >(args)...);
}

// critial
template< typename... Args >
constexpr auto critical(Args&&... args) {
	return detail::make_log_tape<level_enum::critical>(
		std::forward< Args >(args)...
	);
}
template< typename... Args >
constexpr auto C(Args&&... args) {
	return critical(std::forward< Args >(args)...);
}

// trace
template< typename... Args >
constexpr auto trace(Args&&... args) {
	return detail::make_log_tape<level_enum::trace>(
		std::forward< Args >(args)...
	);
}
template< typename... Args >
constexpr auto T(Args&&... args) {
	return trace(std::forward< Args >(args)...);
}

// debug
template< typename... Args >
constexpr auto debug(Args&&... args) {
	return detail::make_log_tape<level_enum::debug>(
		std::forward< Args >(args)...
	);
}
template< typename... Args >
constexpr auto D(Args&&... args) {
	return debug(std::forward< Args >(args)...);
}

// off
template< typename... Args >
constexpr auto off(Args&&... args) {
	return detail::make_log_tape<level_enum::off>(
		std::forward< Args >(args)...
	);
}
template< typename... Args >
constexpr auto O(Args&&... args) {
	return off(std::forward< Args >(args)...);
}

/// set custom tag that will appear inside every printed message to any log
BS_API auto set_custom_tag(std::string tag) -> void;

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

namespace std {

// provide convenience overlaods for aout; implemented in logging.cpp
BS_API blue_sky::log::bs_log& endl(blue_sky::log::bs_log& o);

} // namespace std

