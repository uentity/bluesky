/// @file
/// @author uentity
/// @date 26.06.2020
/// @brief Tree handle engine impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/engine.h>
#include "engine_impl.h"

#include <caf/send.hpp>

#include <memory_resource>

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  weak_ptr
 *-----------------------------------------------------------------------------*/
engine::weak_ptr_base::weak_ptr_base(const engine& src)
	: actor_(src.actor_), pimpl_(src.pimpl_)
{
	if(!src.actor_ || !src.pimpl_) reset();
}

auto engine::weak_ptr_base::assign(const engine& rhs) -> void {
	pimpl_ = rhs.pimpl_;
	actor_ = rhs.actor_;
}

auto engine::weak_ptr_base::lock() const -> engine {
	return engine(actor_.lock(), pimpl_.lock());
}

auto engine::weak_ptr_base::expired() const -> bool {
	return actor_.expired();
}

auto engine::weak_ptr_base::reset() -> void {
	pimpl_.reset();
	actor_.reset();
}

auto engine::weak_ptr_base::operator<(const weak_ptr_base& rhs) const -> bool {
	return pimpl_.owner_before(rhs.pimpl_);
}

auto engine::weak_ptr_base::operator==(const weak_ptr_base& rhs) const -> bool {
	return !(*this < rhs || rhs < *this);
}

auto engine::weak_ptr_base::operator!=(const weak_ptr_base& rhs) const -> bool {
	return !(*this == rhs);
}

auto engine::weak_ptr_base::operator==(const engine& rhs) const -> bool {
	return !(pimpl_.owner_before(rhs.pimpl_) || rhs.pimpl_.owner_before(pimpl_));
}

auto engine::weak_ptr_base::operator!=(const engine& rhs) const -> bool {
	return !(*this == rhs);
}

/*-----------------------------------------------------------------------------
 *  engine
 *-----------------------------------------------------------------------------*/
///////////////////////////////////////////////////////////////////////////////
//  actor_handle
//
struct engine::actor_handle {
	caf::actor core_;

	actor_handle(caf::actor core) : core_(std::move(core)) {}

	// destructor of actor handle terminates wrapped actor
	~actor_handle() {
		caf::anon_send_exit(core_, caf::exit_reason::user_shutdown);
	}
};

// setup synchronized pool allocator for actor handles
static auto impl_pool = std::pmr::synchronized_pool_resource{};
static auto ahdl_alloc = std::pmr::polymorphic_allocator<engine::actor_handle>(&impl_pool);

engine::engine(sp_ahandle ah, sp_engine_impl pimpl) :
	actor_(std::move(ah)), pimpl_(std::move(pimpl))
{}

engine::engine(caf::actor engine_actor, sp_engine_impl pimpl) :
	engine(std::allocate_shared<actor_handle>(ahdl_alloc, std::move(engine_actor)), std::move(pimpl))
{}

auto engine::operator==(const engine& rhs) const -> bool {
	return pimpl_ == rhs.pimpl_;
}

auto engine::operator!=(const engine& rhs) const -> bool {
	return !(*this == rhs);
}

auto engine::operator==(const caf::actor& rhs) const -> bool {
	return raw_actor() == rhs;
}

auto engine::operator<(const engine& rhs) const -> bool {
	return pimpl_.owner_before(rhs.pimpl_);
}

auto engine::has_engine() const noexcept -> bool {
	return pimpl_ && actor_;
}

auto engine::raw_actor() const noexcept -> const caf::actor& {
	return actor_->core_;
}

auto engine::install_raw_actor(caf::actor engine_actor) -> void {
	actor_ = std::make_shared<actor_handle>(std::move(engine_actor));
}

auto engine::hash() const noexcept -> std::size_t {
	return std::hash<engine::sp_engine_impl>{}(pimpl_);
}

auto engine::swap(engine& rhs) noexcept -> void {
	using std::swap;
	swap(actor_, rhs.actor_);
	swap(pimpl_, rhs.pimpl_);
}

auto engine::type_id() const -> std::string_view {
	return pimpl_->type_id();
}

auto engine::home() const -> const caf::group& {
	return pimpl_->home;
}

auto engine::home_id() const -> std::string_view {
	return pimpl_->home_id();
}

auto engine::unsubscribe(std::uint64_t event_cb_id) -> void {
	kernel::radio::bye_actor(event_cb_id);
}

auto engine::unsubscribe() const -> void {
	caf::anon_send(home(), a_bye{});
}

NAMESPACE_END(blue_sky::tree)
