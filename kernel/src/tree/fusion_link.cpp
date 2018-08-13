/// @file
/// @author uentity
/// @date 10.08.2018
/// @brief Fusion link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>
#include <bs/tree/fusion.h>
#include <bs/tree/node.h>

#include <atomic>
#include <future>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  fusion_link_impl
 *-----------------------------------------------------------------------------*/
struct fusion_link::fusion_link_impl {
	fusion_link_impl(sp_fusion&& bridge, sp_node&& data) :
		pop_status_(OpStatus::Void), data_status_(OpStatus::Void),
		bridge_(std::move(bridge)), data_(std::move(data))
	{}

	using op_result = std::pair<error, sp_node>;

	// invoke `fusion_iface::populate()` in sync way
	op_result populate(const std::string& child_type_id = "") {
		// helper that sets populate status depending on result
		auto pop_result = [this](error e) {
			pop_status_ = (e ? OpStatus::Error : OpStatus::OK);
			return e;
		};

		try {
			// invoke `fusion_iface::populate()`
			pop_status_ = OpStatus::Pending;
			return {pop_result(bridge_->populate(data_, child_type_id)), data_};
		}
		catch(const error& e) {
			return {pop_result(e), nullptr};
		}
		catch(const std::exception& e) {
			return {pop_result(e.what()), nullptr};
		}
		catch(...) {
			pop_status_ = OpStatus::Error;
			throw;
		}
	}

	// invoke `fusion_iface::pull_data()` in sync way
	op_result pull_data() {
		// helper that sets populate status depending on result
		auto pop_result = [this](error e) {
			data_status_ = (e ? OpStatus::Error : OpStatus::OK);
			return e;
		};

		try {
			// invoke `fusion_iface::pull_data()`
			data_status_ = OpStatus::Pending;
			return {pop_result(bridge_->pull_data(data_)), data_};
		}
		catch(const error& e) {
			return {pop_result(e), nullptr};
		}
		catch(const std::exception& e) {
			return {pop_result(e.what()), nullptr};
		}
		catch(...) {
			data_status_ = OpStatus::Error;
			throw;
		}
	}

	// populate/data status
	std::atomic<OpStatus> pop_status_, data_status_;
	// bridge
	sp_fusion bridge_;
	// contained object
	sp_node data_;
};

/*-----------------------------------------------------------------------------
 *  fusion_link
 *-----------------------------------------------------------------------------*/
fusion_link::fusion_link(sp_fusion bridge, std::string name, sp_node data, Flags f) :
	link(std::move(name), f),
	pimpl_(std::make_unique<fusion_link_impl>(std::move(bridge), std::move(data)))
{}

fusion_link::fusion_link(
	sp_fusion bridge, std::string name, const char* obj_type, std::string oid, Flags f
) :
	link(std::move(name), f),
	pimpl_(std::make_unique<fusion_link_impl>(
		std::move(bridge), BS_KERNEL.create_object(obj_type, std::move(oid))
	))
{
	if(!pimpl_->data_)
		bserr() << log::E("fusion_link: cannot create object of type '{}'! Empty link!") << obj_type << log::end;
}

fusion_link::~fusion_link() {}

auto fusion_link::clone(bool deep) const -> sp_link {
	auto res = std::make_shared<fusion_link>(
		pimpl_->bridge_, name_,
		deep ? BS_KERNEL.clone_object(std::static_pointer_cast<objbase>(pimpl_->data_)) : pimpl_->data_
	);
	return res;
}

auto fusion_link::type_id() const -> std::string {
	return "fusion_link";
}

auto fusion_link::oid() const -> std::string {
	return pimpl_->data_->id();
}

auto fusion_link::obj_type_id() const -> std::string {
	return pimpl_->data_->type_id();
}

auto fusion_link::data() const -> sp_obj {
	switch(pimpl_->data_status_) {
		case OpStatus::OK :
			return pimpl_->data_;
		case OpStatus::Pending :
			return nullptr;
		default :
			pimpl_->pop_status_ = OpStatus::Pending;
			return pimpl_->pull_data().second;
	}
}

auto fusion_link::data_node() const -> sp_node {
	switch(pimpl_->data_status_) {
		case OpStatus::OK :
			return pimpl_->data_;
		case OpStatus::Pending :
			return nullptr;
		default :
			pimpl_->pop_status_ = OpStatus::Pending;
			return pimpl_->populate().second;
	}
}

auto fusion_link::populate_status() const -> OpStatus {
	return pimpl_->pop_status_;
}

auto fusion_link::reset_populate_status(OpStatus new_status) -> void {
	pimpl_->pop_status_ = new_status;
}

auto fusion_link::data_status() const -> OpStatus {
	return pimpl_->data_status_;
}

auto fusion_link::reset_data_status(OpStatus new_status) -> void {
	pimpl_->data_status_ = new_status;
}

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

