/// @file
/// @author uentity
/// @date 10.08.2018
/// @brief Fusion link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "fusion_link_impl.h"

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

fusion_link::fusion_link(std::string name, sp_fusion bridge, sp_node data, Flags f) :
	link(std::move(name), f),
	pimpl_(std::make_unique<impl>(std::move(bridge), std::move(data)))
{
	// run actor
	pimpl_->actor_ = BS_KERNEL.actor_system().spawn(impl::async_api);
	// connect actor with sender
	pimpl_->init_sender();
}

fusion_link::fusion_link(
	std::string name, sp_fusion bridge, const char* obj_type, std::string oid, Flags f
) :
	link(std::move(name), f),
	pimpl_(std::make_unique<impl>(
		std::move(bridge), BS_KERNEL.create_object(obj_type, std::move(oid))
	))
{
	if(!pimpl_->data_)
		bserr() << log::E("fusion_link: cannot create object of type '{}'! Empty link!") << obj_type << log::end;
}

fusion_link::~fusion_link() {
	//pimpl_->sender()->wait_for(pimpl_->actor());
	std::cout << "<<<<< fusion_link::destructor" << std::endl;
}

auto fusion_link::test() const -> void {
	pimpl_->send(42);
	std::cout << "<<<<< fusion_link::test() message sent" << std::endl;
}

auto fusion_link::clone(bool deep) const -> sp_link {
	auto res = std::make_shared<fusion_link>(
		name(), pimpl_->bridge_,
		deep ? BS_KERNEL.clone_object(std::static_pointer_cast<objbase>(pimpl_->data_)) : pimpl_->data_
	);
	return res;
}

auto fusion_link::type_id() const -> std::string {
	return "fusion_link";
}

auto fusion_link::oid() const -> std::string {
	if(pimpl_->data_)
		return pimpl_->data_->id();
	return {};
}

auto fusion_link::obj_type_id() const -> std::string {
	if(pimpl_->data_)
		return pimpl_->data_->type_id();
	return type_descriptor::nil().name;
}

auto fusion_link::data_ex() const -> result_or_err<sp_obj> {
	return pimpl_->pull_data();
}

auto fusion_link::data_node_ex() const -> result_or_err<sp_node> {
	return pimpl_->populate();
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

auto fusion_link::data(process_data_cb f) const -> void {
	pimpl_->send(
		flnk_data_atom(),
		bs_shared_this<fusion_link>(),
		std::move(f)
	);
}

auto fusion_link::data_node(process_data_cb f) const -> void {
	pimpl_->send(flnk_populate_atom(), bs_shared_this<fusion_link>(), std::move(f), "");
}

auto fusion_link::populate(process_data_cb f, std::string child_type_id) const -> void {
	pimpl_->send(
		flnk_populate_atom(), bs_shared_this<fusion_link>(), std::move(f), std::move(child_type_id)
	);
}

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

