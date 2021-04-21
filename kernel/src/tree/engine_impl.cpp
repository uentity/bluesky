/// @author uentity
/// @date 01.07.2020
/// @brief engine::impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "engine_impl.h"
#include "engine_actor.h"
#include "link_impl.h"
#include "node_impl.h"
#include "nil_engine.h"
#include "../kernel/radio_subsyst.h"

#include <bs/kernel/radio.h>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
namespace kradio = kernel::radio;

[[maybe_unused]] auto adbg_impl(engine_actor_base* A) -> caf::actor_ostream {
	auto os = caf::aout(A);
	os << "[" << A->pimpl_->type_id();
	if(const auto hid = A->pimpl_->home_id(); !hid.empty())
		os << " " << hid;
	return (os << "]: ");
}

/*-----------------------------------------------------------------------------
 *  event
 *-----------------------------------------------------------------------------*/
inline static auto origin_is_nil(const caf::actor& origin) {
	return !origin || nil_link::nil_engine() == origin || nil_node::nil_engine() == origin;
}

auto event::origin_link() const -> link {
	if(origin_is_nil(origin))
		return link{};
	else
		return actorf<engine::sp_engine_impl>(origin, kradio::timeout(true), a_impl{})
		.map([](auto&& eimpl) {
			if(eimpl->type_id() != node_impl::type_id_())
				return std::static_pointer_cast<link_impl>(eimpl)->super_engine();
			else return link{};
		}).value_or(link{});
}

auto event::origin_node() const -> node {
	if(origin_is_nil(origin))
		return node::nil();
	else
		return actorf<engine::sp_engine_impl>(origin, kradio::timeout(true), a_impl{})
		.map([](auto&& eimpl) {
			if(eimpl->type_id() == node_impl::type_id_())
				return std::static_pointer_cast<node_impl>(eimpl)->super_engine();
			else return node::nil();
		}).value_or(node::nil());
}

/*-----------------------------------------------------------------------------
 *  engine_impl
 *-----------------------------------------------------------------------------*/
auto engine::impl::home_id() const -> std::string_view {
	return home ? home.get()->identifier() : std::string_view{};
}

auto engine::impl::swap(impl& rhs) -> void {
	using std::swap;
	swap(home, rhs.home);
}

/*-----------------------------------------------------------------------------
 *  engine_actor
 *-----------------------------------------------------------------------------*/
engine_actor_base::engine_actor_base(caf::actor_config& cfg, caf::group egrp, sp_engine_impl Eimpl) :
	super(cfg), pimpl_(std::move(Eimpl))
{
	// sanity
	if(!pimpl_) throw error{"engine actor: bad (null) impl passed"};
	// remember link's local group
	pimpl_->home = std::move(egrp);
	if(pimpl_->home)
		adbg(this) << "joined self group" << std::endl;

	// exit after kernel
	KRADIO.register_citizen(this);

	// prevent termination in case some errors happens in group members
	// for ex. if they receive unexpected messages (translators normally do)
	set_error_handler([this](caf::error& er) {
		switch(static_cast<caf::sec>(er.code())) {
		case caf::sec::unexpected_message :
		case caf::sec::request_timeout :
		case caf::sec::request_receiver_down :
			break;
		default:
			default_error_handler(this, er);
		}
	});

	// completely ignore unexpected messages without error backpropagation
	set_default_handler(noop_r<caf::message>());
}

auto engine_actor_base::goodbye() -> void {
	adbg(this) << "goodbye" << std::endl;
	auto& home = pimpl_->home;
	if(home) {
		// say goodbye to self group
		send(home, a_bye());
		leave(home);
		adbg(this) << "left self group" << std::endl;
	}
}

auto engine_actor_base::on_exit() -> void {
	adbg(this) << "dies" << std::endl;
	// be polite with everyone
	goodbye();
	// force release strong ref to engine's impl
	pimpl_.reset();

	KRADIO.release_citizen(this);
}

NAMESPACE_END(blue_sky::tree)
