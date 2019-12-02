/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Code shared among different BS actors
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/atoms.h>
#include <bs/actor_common.h>
#include <bs/detail/function_view.h>

#include <caf/stateful_actor.hpp>

#include <boost/uuid/uuid_io.hpp>

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

inline constexpr auto def_data_timeout = timespan{ std::chrono::seconds(3) };
inline const std::string nil_grp_id = "<null>";

/// obtain configured timeout for queries
BS_API auto def_timeout(bool for_data = false) -> caf::duration;

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
