/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief Implementation os link
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "engine_impl.h"
#include "link_actor.h"
#include "nil_engine.h"
#include "node.h"

#include <bs/log.h>
#include <bs/kernel/radio.h>

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
// [NOTE] the purpose of link constructors is to ALWAYS produce a valid link
// 'valid' means that `pimpl()` returns non-null value, `raw_actor()` returns valid handle
// at worst link may become 'nil link' that is also a valid link
link::link(engine&& e) :
	engine(std::move(e))
{
	if(!has_engine())
		*this = nil_link::nil_engine();
}

link::link() :
	engine(nil_link::nil_engine())
{}

link::link(sp_engine_impl impl, bool start_actor) :
	engine(nil_link::actor(), std::move(impl))
{
	if(start_actor) start_engine();
}

link::link(const link& rhs, std::string_view rhs_type_id) :
	link(rhs)
{
	if(type_id() != rhs_type_id) reset();
}

link::link(std::string name, sp_obj data, Flags f) :
	link(hard_link(std::move(name), std::move(data), f))
{}

link::link(std::string name, node folder, Flags f) :
	link(hard_link(std::move(name), std::move(folder), f))
{}

auto link::make_root_(engine donor) -> link {
	auto res = link(std::move(donor));
	if(res) res.pimpl()->propagate_handle();
	return res;
}

auto link::is_nil() const -> bool {
	return pimpl_ == nil_link::pimpl();
}

auto link::reset() -> void {
	if(!is_nil())
		*this = nil_link::nil_engine();
}

auto link::pimpl() const -> link_impl* {
	return static_cast<link_impl*>(pimpl_.get());
}

auto link::start_engine() -> bool {
	if(actor_ == nil_link::actor()) {
		install_raw_actor(pimpl()->spawn_actor(std::static_pointer_cast<link_impl>(pimpl_)));
		// explicitly setup weak link from pimpl to engine
		pimpl()->reset_super_engine(*this);
		return true;
	}
	return false;
}

auto link::clone(bool deep) const -> link {
	return { pimpl()->clone(deep) };
}

auto link::id() const -> lid_type {
	return pimpl()->id_;
}

auto link::rename(std::string new_name) const -> bool {
	return pimpl()->actorf<std::size_t>(
		*this, a_lnk_rename(), std::move(new_name)
	).value_or(0);
}

auto link::rename(launch_async_t, std::string new_name) const -> void {
	caf::anon_send(actor(), a_lnk_rename(), std::move(new_name));
}

auto link::owner() const -> node {
	return pimpl()->owner();
}

auto link::info() const -> result_or_err<inode> {
	return pimpl()->actorf<result_or_errbox<inodeptr>>(*this, a_lnk_inode())
	.and_then([](const inodeptr& i) {
		return i ?
			result_or_err<inode>(*i) :
			tl::make_unexpected(error::quiet(Error::EmptyInode));
	});
}

auto link::info(unsafe_t) const -> result_or_err<inode> {
	return pimpl()->get_inode()
	.and_then([](const inodeptr& i) {
		return i ?
			result_or_err<inode>(*i) :
			tl::make_unexpected(error::quiet(Error::EmptyInode));
	});
}

auto link::flags() const -> Flags {
	return pimpl()->actorf<Flags>(*this, a_lnk_flags()).value_or(Flags::Plain);
}

auto link::flags(unsafe_t) const -> Flags {
	return pimpl()->flags_;
}

auto link::set_flags(Flags new_flags) const -> void {
	caf::anon_send(actor(), a_lnk_flags(), new_flags);
}

auto link::req_status(Req request) const -> ReqStatus {
	return pimpl()->req_status(request);
}

auto link::rs_reset(Req request, ReqStatus new_rs) const -> ReqStatus {
	return pimpl()->rs_reset(
		request, ReqReset::Always, new_rs, ReqStatus::Void,
		[&](auto Req, auto new_rs, auto prev_rs) {
			caf::anon_send<high_prio>(pimpl()->actor(*this), a_lnk_status(), request, new_rs, prev_rs);
		}
	);
}

auto link::rs_reset_if_eq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl()->rs_reset(
		request, ReqReset::IfEq, new_rs, self,
		[&](auto Req, auto new_rs, auto prev_rs) {
			caf::anon_send<high_prio>(pimpl()->actor(*this), a_lnk_status(), request, new_rs, prev_rs);
		}
	);
}

auto link::rs_reset_if_neq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl()->rs_reset(
		request, ReqReset::IfNeq, new_rs, self,
		[&](auto Req, auto new_rs, auto prev_rs) {
			caf::anon_send<high_prio>(pimpl()->actor(*this), a_lnk_status(), request, new_rs, prev_rs);
		}
	);
}

///////////////////////////////////////////////////////////////////////////////
//  sync API
//
/// obtain link's human-readable name
auto link::name() const -> std::string {
	return pimpl()->actorf<std::string>(*this, a_lnk_name()).value_or("");
}

auto link::name(unsafe_t) const -> std::string {
	return pimpl()->name_;
}

auto link::oid() const -> std::string {
	return pimpl()->actorf<std::string>(*this, a_lnk_oid())
		.value_or(nil_oid);
}

auto link::oid(unsafe_t) const -> std::string {
	if(auto obj = pimpl()->data(unsafe))
		return obj->id();
	return nil_oid;
}

auto link::obj_type_id() const -> std::string {
	return pimpl()->actorf<std::string>(*this, a_lnk_otid())
		.value_or( nil_otid );
}

auto link::obj_type_id(unsafe_t) const -> std::string {
	if(auto obj = pimpl()->data(unsafe))
		return obj->type_id();
	return nil_otid;
}

auto link::data_ex(bool wait_if_busy) const -> obj_or_err {
	return pimpl()->actorf<obj_or_errbox>(
		long_op, *this, a_data(), wait_if_busy
	);
}

auto link::data(unsafe_t) const -> sp_obj {
	return pimpl()->data(unsafe);
}

auto link::data_node_ex(bool wait_if_busy) const -> node_or_err {
	return pimpl()->actorf<node_or_errbox>(
		long_op, *this, a_data_node(), wait_if_busy
	);
}

auto link::data_node() const -> node {
	return data_node_ex().value_or(node::nil());
}

auto link::data_node(unsafe_t) const -> node {
	if(auto obj = pimpl()->data(unsafe))
		return obj->data_node();
	return node::nil();
}

auto link::data_node_hid() const -> result_or_err<std::string> {
	// [TODO] enable this more efficient path later
	//return pimpl()->actorf<result_or_errbox<std::string>>(*this, a_node_gid());
	return data_node_ex().map([](const node& N) { return N.home_id(); });
}

auto link::data_node_hid(unsafe_t) const -> std::string {
	if(auto me = data_node(unsafe))
		return me.home().get()->identifier();
	return {};
}

auto link::is_node() const -> bool {
	return !data_node_hid().value_or("").empty();
}

///////////////////////////////////////////////////////////////////////////////
//  apply
//
template<bool AsyncApply = false>
static auto make_apply_impl(const link& L, obj_transaction m, bool silent) {
return [=, wL = link::weak_ptr(L), m = std::move(m)](obj_or_errbox obj) mutable {
	auto finally = [=](tr_result&& r) {
		// set status after modificator invoked
		if(!silent)
			L.rs_reset(Req::Data, r ? ReqStatus::OK : ReqStatus::Error);
		return std::move(r);
	};

	// deliver error if couldn't obtain link's data
	if(!obj) {
		auto er = finally(obj.error());
		if constexpr(!AsyncApply) return er;
	}

	// put modificator into object's queue
	if constexpr(AsyncApply) {
		(*obj)->apply(
			launch_async,
			[m = std::move(m), finally = std::move(finally)](sp_obj obj) -> tr_result {
				return finally(tr_eval(std::move(m), std::move(obj)));
			}
		);
	}
	else
		return finally( (*obj)->apply(std::move(m)) );
};
}

auto link::data_apply(obj_transaction m, bool silent) const -> tr_result {
	return make_apply_impl(*this, std::move(m), silent)( data_ex(true) );
}

auto link::apply(simple_transaction tr) const -> error {
	return pimpl()->actorf<error>(*this, a_apply(), std::move(tr));
}

auto link::apply(link_transaction tr) const -> error {
	return pimpl()->actorf<error>(*this, a_apply(), std::move(tr));
}

auto link::apply(launch_async_t, simple_transaction tr) const -> void {
	caf::anon_send(pimpl()->actor(*this), a_apply(), std::move(tr));
}

auto link::apply(launch_async_t, link_transaction tr) const -> void {
	caf::anon_send(pimpl()->actor(*this), a_apply(), std::move(tr));
}

///////////////////////////////////////////////////////////////////////////////
//  async API
//
auto link::data(process_data_cb f, bool high_priority) const -> void {
	anon_request(
		actor(), kernel::radio::timeout(true), high_priority,
		[f = std::move(f), wself = weak_ptr(*this)](obj_or_errbox eobj) {
			if(auto self = wself.lock())
				f(std::move(eobj), std::move(self));
		},
		a_data(), true
	);
}

auto link::data_node(process_dnode_cb f, bool high_priority) const -> void {
	anon_request(
		actor(), kernel::radio::timeout(true), high_priority,
		[f = std::move(f), wself = weak_ptr(*this)](node_or_errbox enode) {
			if(auto self = wself.lock())
				f(std::move(enode), std::move(self));
		},
		a_data_node(), true
	);
}

auto link::data_apply(launch_async_t, obj_transaction m, bool silent) const -> void {
	anon_request(
		actor(), kernel::radio::timeout(true), false,
		make_apply_impl<true>(*this, std::move(m), silent),
		a_data(), true
	);
}

NAMESPACE_END(blue_sky::tree)
