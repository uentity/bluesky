/// @file
/// @author uentity
/// @date 01.07.2020
/// @brief engine::impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "engine_impl.h"
#include "engine_actor.h"
#include "../kernel/radio_subsyst.h"

#include <bs/kernel/radio.h>

#include <memory_resource>

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
 *  engine_impl
 *-----------------------------------------------------------------------------*/
static auto factor_pool = std::pmr::synchronized_pool_resource{};
static auto factor_alloc = std::pmr::polymorphic_allocator<caf::scoped_actor>(&factor_pool);

auto engine::impl::factor(const engine* L) -> sp_scoped_actor {
	// check if elem is already inserted and find insertion position
	{
		auto guard = guard_.lock(detail::shared);
		if(auto pf = rpool_.find(L); pf != rpool_.end())
			return pf->second;
	}
	// make insertion
	auto mguard = guard_.lock();
	return rpool_.try_emplace(
		L, std::allocate_shared<caf::scoped_actor>(factor_alloc, kradio::system())
	).first->second;
}

auto engine::impl::release_factor(const engine* L) -> void {
	auto guard = guard_.lock();
	rpool_.erase(L);
}

auto engine::impl::release_factors() -> void {
	auto guard = guard_.lock();
	rpool_.clear();
}

auto engine::impl::home_id() const -> std::string_view {
	return home ? home.get()->identifier() : std::string_view{};
}

auto engine::impl::swap(impl& rhs) -> void {
	using std::swap;
	swap(home, rhs.home);
	swap(rpool_, rhs.rpool_);
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
	set_error_handler([this](caf::error er) {
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
	set_default_handler([](auto*, auto&) -> caf::result<caf::message> {
		return caf::none;
	});
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
	pimpl_->release_factors();
	pimpl_.reset();

	KRADIO.release_citizen(this);
}

NAMESPACE_END(blue_sky::tree)
