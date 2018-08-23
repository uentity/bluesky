/// @file
/// @author uentity
/// @date 15.08.2018
/// @brief Impl part of fusion_link PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/kernel.h>
#include <bs/tree/fusion.h>
#include <bs/tree/node.h>
#include <bs/atoms.h>
#include <bs/detail/async_api_mixin.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <caf/all.hpp>
#include <mutex>
//#include <atomic>
#include <thread>

#include <iostream>

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::fusion_link::process_data_cb)

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  fusion_link::impl
 *-----------------------------------------------------------------------------*/
struct fusion_link::impl : public detail::async_api_mixin<fusion_link::impl> {
	// populate/data status
	//std::atomic<OpStatus> pop_status_, data_status_;
	OpStatus pop_status_, data_status_;
	// bridge
	sp_fusion bridge_;
	// contained object
	sp_node data_;
	// sync access to object
	std::mutex sync_data_;

	// ctor
	impl(sp_fusion&& bridge, sp_node&& data) :
		pop_status_(OpStatus::Void), data_status_(OpStatus::Void),
		bridge_(std::move(bridge)), data_(std::move(data))
	{
	}

	// result of fusion operation
	using op_result = result_or_err<sp_node>;

	// invoke any fusion operation F
	template<typename F, typename... Args>
	op_result fusion_call(OpStatus& status, F f, Args&&... args) {
		// helper that sets populate status depending on result
		static const auto pop_result = [&status](error e) {
			status = (e ? OpStatus::Error : OpStatus::OK);
			return e;
		};
		// sanity
		if(!bridge_) return tl::make_unexpected(error::quiet("Bad fusion bridge"));

		// lock link state
		std::lock_guard<std::mutex> guard(sync_data_);

		try {
			switch(status) {
			case OpStatus::OK :
				return data_;
			case OpStatus::Busy :
				return tl::make_unexpected(error::quiet("fusion_link is busy"));
			default :
				// invoke operation
				status = OpStatus::Busy;
				auto err = pop_result( (bridge_.get()->*f)(data_, std::forward<Args>(args)...) );
				return err.ok() ? op_result(data_) : tl::make_unexpected(std::move(err));
			}
		}
		catch(const error& e) {
			return tl::make_unexpected( pop_result(e) );
		}
		catch(const std::exception& e) {
			return tl::make_unexpected( pop_result(e.what()) );
		}
		catch(...) {
			return tl::make_unexpected( pop_result(error()) );
		}
	}

	// invoke `fusion_iface::populate()` in sync way
	op_result populate(const std::string& child_type_id = "") {
		return fusion_call(pop_status_, &fusion_iface::populate, child_type_id);
	}

	// invoke `fusion_iface::pull_data()` in sync way
	op_result pull_data() {
		return fusion_call(data_status_, &fusion_iface::pull_data);
	}

	///////////////////////////////////////////////////////////////////////////////
	//  async API
	//
	// actor type for async API
	using actor_t = caf::typed_actor<
		caf::reacts_to<flnk_data_atom, sp_cfusion_link, fusion_link::process_data_cb>,
		caf::reacts_to<flnk_populate_atom, sp_cfusion_link, fusion_link::process_data_cb, std::string>,
		caf::reacts_to<int>
	>;

	// async API actor handle
	actor_t actor_;
	auto actor() const -> const actor_t& { return actor_; }

	// behaviour
	static auto async_api(actor_t::pointer self) -> actor_t::behavior_type {
		using cb_arg = result_or_err<sp_clink>;

		self->set_exit_handler([](caf::exit_msg&) {
			std::cout << "******* fusion_link::actor down ***********" << std::endl;
		});
		return {
			[](flnk_data_atom, sp_cfusion_link lnk, const process_data_cb& f) {
				auto res = lnk->pimpl_->pull_data();
				f( res ? cb_arg(std::move(lnk)) : tl::make_unexpected(res.error()) );
				// alternatively I can do this:
				//res.map([&f, lnk(std::move(lnk))](const auto&) { f(std::move(lnk)); });
				//res.map_error([&f](auto&& err) { f(tl::make_unexpected(err)); });
				// more functional-like style, but I think one-liner function call looks better
			},
			[](
				flnk_populate_atom, sp_cfusion_link lnk,
				const process_data_cb& f, const std::string& obj_type
			) {
				auto res = lnk->pimpl_->populate(obj_type);
				f( res ? cb_arg(std::move(lnk)) : tl::make_unexpected(res.error()) );
			},
			[](int) {
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(3s);
				std::cout << "<<<<<< fusion_link::actor waked" << std::endl;
			}
		};
	}

	~impl() {
		//sender()->wait_for(actor());
		std::cout << "<<<<< fusion_link::pimpl::destructor" << std::endl;
	}
};

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

