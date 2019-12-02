/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Code shared among different BS actors
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>
#include <bs/timetypes.h>
#include <bs/error.h>
#include <bs/atoms.h>
#include <bs/kernel/radio.h>
#include <bs/detail/function_view.h>

#include <caf/fwd.hpp>
#include <caf/function_view.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/stateful_actor.hpp>

#include <boost/uuid/uuid_io.hpp>

#include <optional>

#define OMIT_OBJ_SERIALIZATION                                                          \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::error::box)                                   \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::sp_obj)                                       \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::tree::inodeptr)                               \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::tree::sp_link)                                \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::tree::sp_node)                                \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::result_or_errbox<::blue_sky::sp_obj>)         \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::result_or_errbox<::blue_sky::tree::inodeptr>) \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::result_or_errbox<::blue_sky::tree::sp_link>)  \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::result_or_errbox<::blue_sky::tree::sp_node>)

NAMESPACE_BEGIN(blue_sky::tree)

inline constexpr auto high_prio = caf::message_priority::high;
inline constexpr auto def_data_timeout = timespan{ std::chrono::seconds(3) };
inline const std::string nil_grp_id = "<null>";

/// obtain configured timeout for queries
BS_API auto def_timeout(bool for_data = false) -> caf::duration;

BS_API auto forward_caf_error(const caf::error& er) -> error;

/// blocking invoke actor & return response like a function
/// [NOTE] always return `result_or_errbox<R>`
template<typename R, typename Actor, typename... Args>
inline auto actorf(caf::function_view<Actor>& factor, Args&&... args) {
	auto x = factor(std::forward<Args>(args)...);

	// try extract value
	using x_value_t = typename decltype(x)::value_type;
	const auto extract_value = [&](auto& res) {
		// caf err passtrough
		if(!x) {
			res.emplace(tl::make_unexpected( forward_caf_error(x.error()) ));
			return;
		}

		if constexpr(std::is_same_v<x_value_t, caf::message>)
			x->extract({ [&](R value) {
				res.emplace(std::move(value));
			} });
		else
			res.emplace(std::move(*x));
		if(!res) res.emplace( tl::make_unexpected(error{ "actorf: wrong result type R specified" }) );
	};

	if constexpr(tl::detail::is_expected<R>::value) {
		std::optional<R> res;
		extract_value(res);
		return std::move(*res);
	}
	else {
		std::optional<result_or_err<R>> res;
		extract_value(res);
		return std::move(*res);
	}
}

/// constructs function_view inside from passed handle & timeout
template<typename R, typename H, typename... Args>
inline auto actorf(const H& handle, blue_sky::timespan timeout, Args&&... args) {
	return actorf<R>(
		caf::make_function_view(handle, caf::duration{ timeout }), std::forward<Args>(args)...
	);
}

/// spawn temp actor that makes specified request to `A` and pass result to callback `f`
//template<caf::spawn_options Os = caf::no_spawn_options, typename Actor, typename F, typename... Args>
//auto anon_request(Actor A, caf::duration timeout, bool high_priority, F f, Args&&... args) -> void {
//	kernel::radio::system().spawn<Os>([f = std::move(f)] (
//		caf::event_based_actor* self, Actor A, caf::duration t, bool high_priority, Args&&... a_args
//	) mutable -> caf::behavior {
//		auto req = high_priority ?
//			self->request<caf::message_priority::high>(A, t, std::forward<Args>(a_args)...) :
//			self->request<caf::message_priority::normal>(A, t, std::forward<Args>(a_args)...);
//		req.then(std::move(f));
//
//		return {};
//	}, std::move(A), timeout, high_priority, std::forward<Args>(args)...);
//}

template<caf::spawn_options Os = caf::no_spawn_options, typename Actor, typename F, typename... Args>
auto anon_request(Actor A, caf::duration timeout, bool high_priority, F f, Args&&... args) -> void {
	kernel::radio::system().spawn<Os>([
		high_priority, f = std::move(f), A = std::move(A), t = std::move(timeout),
		args = std::make_tuple(std::forward<Args>(args)...)
	] (caf::event_based_actor* self) mutable -> caf::behavior {
		std::apply([self, high_priority, A = std::move(A), t = std::move(t)](auto&&... args) {
			return high_priority ?
				self->request<caf::message_priority::high>(A, t, std::forward<decltype(args)>(args)...) :
				self->request<caf::message_priority::normal>(A, t, std::forward<decltype(args)>(args)...);
		}, std::move(args))
		.then(std::move(f));

		return {};
	});
}

/// listens to event from specified group, execute given character & self-detruct on `a_bye` message
template<typename State>
auto ev_listener_actor(
	caf::stateful_actor<State>* self, caf::group tgt_grp,
	std::function<caf::message_handler (caf::stateful_actor<State>*)> make_character
) -> caf::behavior {
	// silently drop all other messages not in my character
	self->set_default_handler(caf::drop);
	// come to mummy
	self->join(tgt_grp);
	auto& Reg = kernel::radio::system().registry();
	Reg.put(self->id(), self);

	// unsubscribe when parent leaves its group
	auto C = make_character(self);
	return caf::message_handler{
		[self, grp = std::move(tgt_grp), &Reg, C](a_bye) mutable {
			self->leave(grp);
			Reg.erase(self->id());
			// invoke custom handler from character
			auto bye_msg = caf::make_message(a_bye());
			C(bye_msg);
		}
	}.or_else(std::move(C));

	//return make_character(self).or_else(
	//	[self, grp = std::move(tgt_grp), &Reg](a_bye) {
	//		self->leave(grp);
	//		Reg.erase(self->id());
	//	}
	//);
}

NAMESPACE_END(blue_sky::tree)
