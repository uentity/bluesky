/// @file
/// @author uentity
/// @date 20.11.2017
/// @brief Symbolic link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/tree/tree.h>
#include <bs/tree/errors.h>
#include <bs/kernel/config.h>
#include <bs/kernel/tools.h>

#include "link_actor.h"

NAMESPACE_BEGIN(blue_sky::tree)
///////////////////////////////////////////////////////////////////////////////
//  sym_link actor
//
struct sym_link_actor : public link_actor {
	using super = link_actor;

	using super::super;

	auto target() const -> link_or_err {
		return static_cast<sym_link_impl&>(impl).target();
	}

	template<typename R, typename... Args>
	auto delegate_target(Args&&... args)
	-> std::enable_if_t<tl::detail::is_expected<R>::value, caf::result<R>> {
		if(auto p = target())
			return delegate(link::actor(p.value()), std::forward<Args>(args)...);
		else
			return R{ tl::unexpect, p.error() };
	}

	template<typename R, typename... Args>
	auto delegate_target(R errval, Args&&... args)
	-> std::enable_if_t<!tl::detail::is_expected<R>::value, caf::result<R>> {
		if(auto p = target())
			return delegate(link::actor(p.value()), std::forward<Args>(args)...);
		else
			return errval;
	}

	// delegate name, OID, etc requests to source link
	auto make_behavior() -> caf::behavior override {
		return caf::message_handler({

			[this](a_lnk_oid) -> caf::result<std::string> {
				return delegate_target<std::string>(nil_otid, a_lnk_oid());
			},

			[this](a_lnk_otid) -> caf::result<std::string> {
				return delegate_target<std::string>(nil_otid, a_lnk_otid());
			},

			[this](a_home_id) -> caf::result< std::string > {
				return delegate_target< std::string >(nil_oid, a_home_id());
			},

			[this](a_lnk_inode) -> caf::result< result_or_errbox<inodeptr> > {
				return delegate_target< result_or_errbox<inodeptr> >(a_lnk_inode());
			},

		}).or_else(super::make_behavior());
	}
};

///////////////////////////////////////////////////////////////////////////////
//  sym_link impl
//
sym_link_impl::sym_link_impl(std::string name, std::string path, Flags f)
	: super(std::move(name), f), path_(std::move(path))
{}

sym_link_impl::sym_link_impl()
	: super()
{}

auto sym_link_impl::target() const -> link_or_err {
	const auto parent = owner();
	if(!parent) return unexpected_err_quiet(Error::UnboundSymLink);

	link src_link;
	if(auto er = error::eval_safe([&] { src_link = deref_path(path_, parent); }); er)
		return tl::make_unexpected(std::move(er));
	else if(src_link)
		return src_link;
	return unexpected_err_quiet(Error::LinkExpired);
}

auto sym_link_impl::data() -> obj_or_err {
	auto res = target().and_then([](const link& src_link) {
		return src_link.data_ex();
	});
#if defined(_DEBUG)
	if(!res) {
		auto& er = res.error();
		std::cout << ">>> " << to_string(id_) << ' ' << er.what() << std::endl;
		//std::cout << kernel::tools::get_backtrace(16) << std::endl;
	}
#endif
	return res;
}

auto sym_link_impl::data(unsafe_t) const -> sp_obj {
	return target().map([](const link& src_link) {
		return src_link.data(unsafe);
	}).value_or(nullptr);
}

auto sym_link_impl::spawn_actor(sp_limpl limpl) const -> caf::actor {
	return spawn_lactor<sym_link_actor>(std::move(limpl));
}

auto sym_link_impl::clone(link_actor*, bool deep) const -> caf::result<sp_limpl> {
	// no deep copy support for symbolic link
	return std::make_shared<sym_link_impl>(name_, path_, flags_);
}

auto sym_link_impl::propagate_handle() -> node_or_err {
	// sym link cannot be a node's handle and also must make tree query here, so just return error
	return unexpected_err_quiet(Error::EmptyData);
}

///////////////////////////////////////////////////////////////////////////////
//  class
//

#define SIMPL static_cast<sym_link_impl&>(*super::pimpl())

/// ctor -- pointee is specified by string path
sym_link::sym_link(std::string name, std::string path, Flags f)
	: link(std::make_shared<sym_link_impl>(std::move(name), std::move(path), f))
{}
/// ctor -- pointee is specified directly - absolute path will be stored
sym_link::sym_link(std::string name, const link& src, Flags f)
	: sym_link(std::move(name), abspath(src), f)
{}

sym_link::sym_link()
	: super(std::make_shared<sym_link_impl>(), false)
{}

bool sym_link::check_alive() {
	auto res = bool(deref_path(SIMPL.path_, owner()));
	auto S = res ? ReqStatus::OK : ReqStatus::Error;
	rs_reset_if_neq(Req::Data, S, S);
	return res;
}

auto sym_link::target() const -> link_or_err {
	return SIMPL.target();
}

/// return stored pointee path
std::string sym_link::target_path(bool human_readable) const {
	auto res = std::string(SIMPL.path_);
	if(human_readable) {
		if(const auto parent = owner())
			res = convert_path(res, parent.handle());
	}
	return res;
}

LINK_CONVERT_TO(sym_link)
LINK_TYPE_DEF(sym_link, sym_link_impl, "sym_link")

NAMESPACE_END(blue_sky::tree)
