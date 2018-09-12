/// @file
/// @author uentity
/// @date 14.08.2018
/// @brief Link-related implementation details
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/link.h>
#include <bs/tree/node.h>
#include <bs/atoms.h>
#include <bs/detail/async_api_mixin.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <caf/all.hpp>
#include <boost/smart_ptr/detail/yield_k.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::link::process_data_cb)

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

using id_type = link::id_type;
using Flags = link::Flags;

namespace {

// global random UUID generator for BS links
static boost::uuids::random_generator gen;

// microsleep until flag is released by someone else
inline auto raise_atomic_flag(std::atomic_flag& flag) -> void {
	for(unsigned k = 0; flag.test_and_set(std::memory_order_acquire); ++k)
		boost::detail::yield(k);
}

inline auto drop_atomic_flag(std::atomic_flag& flag) {
	flag.clear(std::memory_order_release);
}

// RAII raise-drop flag
struct scope_atomic_flag {
	std::atomic_flag& f_;

	explicit scope_atomic_flag(std::atomic_flag& flag)
		: f_(flag)
	{
		raise_atomic_flag(f_);
	}

	~scope_atomic_flag() {
		drop_atomic_flag(f_);
	}
};

using Req = link::Req;
using ReqStatus = link::ReqStatus;

template<typename L, typename F>
static auto link_invoke(
	L* lnk, F f, ReqStatus& status, std::atomic_flag& status_flag
) -> decltype(f(lnk)) {
	//using req_result = decltype((lnk->*f)(std::forward<Args>(args)...));

	bool flag_set = false;

	// helper that captures the status flag
	const auto lock_status = [&status_flag, &flag_set] {
		raise_atomic_flag(status_flag);
		flag_set = true;
	};
	// helper that release flag only if it was set before
	const auto unlock_status = [&status_flag, &flag_set] {
		if(flag_set) {
			status_flag.clear(std::memory_order_release);
			flag_set = false;
		}
	};
	// always release status flag on exit
	const auto release_on_exit = make_scope_guard(unlock_status);

	// helper that sets populate status depending on result
	static const auto set_status = [&status](error e) {
		status = (e ? ReqStatus::Error : ReqStatus::OK);
		return e;
	};

	// 0. capture the flag
	lock_status();

	// 1. invoke link::f
	try {
		switch(status) {
		case ReqStatus::Busy :
			return tl::make_unexpected(error::quiet("link is busy"));
		case ReqStatus::Void :
		case ReqStatus::Error :
			// set Busy status (OK status is not affected)
			status = ReqStatus::Busy;
		default :
			// release status flag before possibly long operation
			unlock_status();
			// invoke operation
			auto res = f(lnk);
			// set flag depending on result
			lock_status();
			status = res ?
				res.value() ?
					ReqStatus::OK :
					ReqStatus::Void :
				ReqStatus::Error;
			// and return
			return res;
			// [TODO] we have to notify parent node that content changed
		}
	}
	catch(const error& e) {
		return tl::make_unexpected( set_status(e) );
	}
	catch(const std::exception& e) {
		return tl::make_unexpected( set_status(e.what()) );
	}
	catch(...) {
		return tl::make_unexpected( set_status(error()) );
	}
}

} // eof hidden namespace

/*-----------------------------------------------------------------------------
 *  link::impl
 *-----------------------------------------------------------------------------*/
struct link::impl : public detail::async_api_mixin<link::impl> {
	id_type id_;
	std::string name_;
	Flags flags_;
	/// contains link's metadata
	inode inode_;
	/// owner node
	std::weak_ptr<node> owner_;
	/// status of operations
	ReqStatus status_[2] = {ReqStatus::Void, ReqStatus::Void};
	mutable std::atomic_flag status_flag_[2] = {ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT};
	// sync access to link's essentail data
	std::mutex solo_;

	impl(std::string&& name, Flags f)
		: id_(gen()), name_(std::move(name)), flags_(f)
	{}

	auto rename_silent(std::string&& new_name) -> void {
		solo_.lock();
		name_ = std::move(new_name);
		solo_.unlock();
	}

	auto rename(std::string&& new_name) -> void {
		rename_silent(std::move(new_name));
		// [TODO] send message instead
		if(auto O = owner_.lock()) {
			O->on_rename(id_);
		}
	}

	auto req_status(Req request) const -> ReqStatus {
		const auto i = (unsigned)request;
		if(i < 2) {
			// atomic read
			auto S = scope_atomic_flag(status_flag_[i]);
			return status_[i];
		}
		return ReqStatus::Void;
	}

	auto rs_reset(Req request, ReqStatus new_rs) {
		const auto i = (unsigned)request;
		if(i >= 2) return ReqStatus::Error;

		// atomic set value
		auto S = scope_atomic_flag(status_flag_[i]);
		const auto self = status_[i];
		status_[i] = new_rs;
		return self;
	}

	auto rs_reset_if_eq(Req request, ReqStatus self_rs, ReqStatus new_rs) {
		const auto i = (unsigned)request;
		if(i >= 2) return ReqStatus::Error;

		// atomic set value
		auto S = scope_atomic_flag(status_flag_[i]);
		const auto self = status_[i];
		if(status_[i] == self_rs) status_[i] = new_rs;
		return self;
	}

	auto rs_reset_if_neq(Req request, ReqStatus self_rs, ReqStatus new_rs) {
		const auto i = (unsigned)request;
		if(i >= 2) return ReqStatus::Error;

		// atomic set value
		auto S = scope_atomic_flag(status_flag_[i]);
		const auto self = status_[i];
		if(status_[i] != self_rs) status_[i] = new_rs;
		return self;
	}

	///////////////////////////////////////////////////////////////////////////////
	//  async API
	//
	// actor type for async API
	using actor_t = caf::typed_actor<
		caf::reacts_to<lnk_data_atom, sp_clink, link::process_data_cb>,
		caf::reacts_to<lnk_dnode_atom, sp_clink, link::process_data_cb>,
		caf::reacts_to<int>
	>;

	// async API actor handle
	actor_t actor_;
	auto actor() const -> const actor_t& { return actor_; }

	// behaviour
	static auto async_api(actor_t::pointer self) -> actor_t::behavior_type {
		using cb_arg = result_or_err<sp_clink>;

		self->set_exit_handler([](caf::exit_msg&) {
			std::cout << "******* link::actor down ***********" << std::endl;
		});
		return {
			[](lnk_data_atom, const sp_clink& lnk, const process_data_cb& f) {
				f(lnk->data_ex(), lnk);
			},
			[](lnk_dnode_atom, const sp_clink& lnk, const process_data_cb& f) {
				f(lnk->data_node_ex(), lnk);
			},
			[](int) {
				using namespace std::chrono_literals;
				std::cout << "<<<<<< link::impl::actor test" << std::endl;
				std::this_thread::sleep_for(3s);
			}
		};
	}

	~impl() {
		//sender()->wait_for(actor());
		std::cout << "<<<<< link::impl::destructor" << std::endl;
	}

};

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

