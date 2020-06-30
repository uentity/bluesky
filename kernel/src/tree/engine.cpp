/// @file
/// @author uentity
/// @date 26.06.2020
/// @brief Tree handle engine impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/engine.h>

#include <caf/send.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  weak_ptr
 *-----------------------------------------------------------------------------*/
engine::weak_ptr_base::weak_ptr_base(const sp_ahandle& ah, const sp_engine_impl& pimpl)
	: actor_(ah), pimpl_(pimpl)
{
	if(!ah || !pimpl) reset();
}

engine::weak_ptr_base::weak_ptr_base(const engine& src)
	: weak_ptr_base(src.actor_, src.pimpl_)
{}

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

auto engine::weak_ptr_base::operator==(const weak_ptr_base& rhs) const -> bool {
	return pimpl_.lock() == rhs.pimpl_.lock();
}

auto engine::weak_ptr_base::operator!=(const weak_ptr_base& rhs) const -> bool {
	return !(*this == rhs);
}

auto engine::weak_ptr_base::operator==(const engine& rhs) const -> bool {
	return pimpl_.lock() == rhs.pimpl_;
}

auto engine::weak_ptr_base::operator!=(const engine& rhs) const -> bool {
	return !(*this == rhs);
}

auto engine::weak_ptr_base::operator<(const weak_ptr_base& rhs) const -> bool {
	return pimpl_.owner_before(rhs.pimpl_);
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

engine::engine(sp_ahandle ah, sp_engine_impl pimpl) :
	actor_(std::move(ah)), pimpl_(std::move(pimpl))
{}

engine::engine(caf::actor engine_actor, sp_engine_impl pimpl) :
	engine(std::make_shared<actor_handle>(std::move(engine_actor)), std::move(pimpl))
{}

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
	std::swap(actor_, rhs.actor_);
	std::swap(pimpl_, rhs.pimpl_);
}

NAMESPACE_END(blue_sky::tree)
