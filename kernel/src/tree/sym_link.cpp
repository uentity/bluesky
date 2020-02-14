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

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include "link_actor.h"

OMIT_OBJ_SERIALIZATION

NAMESPACE_BEGIN(blue_sky::tree)
///////////////////////////////////////////////////////////////////////////////
//  actor + impl
//

// actor
struct sym_link_actor : public link_actor {
	using super = link_actor;

	using super::super;

	auto pointee() const -> result_or_err<link> {
		return static_cast<sym_link_impl&>(impl).pointee();
	}

	template<typename R, typename... Args>
	auto delegate_source(Args&&... args)
	-> std::enable_if_t<tl::detail::is_expected<R>::value, caf::result<R>> {
		if(auto p = pointee())
			return delegate(link::actor(p.value()), std::forward<Args>(args)...);
		else
			return R{ tl::unexpect, p.error() };
	}

	template<typename R, typename... Args>
	auto delegate_source(R errval, Args&&... args)
	-> std::enable_if_t<!tl::detail::is_expected<R>::value, caf::result<R>> {
		if(auto p = pointee())
			return delegate(link::actor(p.value()), std::forward<Args>(args)...);
		else
			return errval;
	}

	// delegate name, OID, etc requests to source link
	auto make_behavior() -> caf::behavior override {
		return caf::message_handler({

			[this](a_lnk_oid) -> caf::result<std::string> {
				return delegate_source<std::string>(nil_otid, a_lnk_oid());
			},

			[this](a_lnk_otid) -> caf::result<std::string> {
				return delegate_source<std::string>(nil_otid, a_lnk_otid());
			},

			[this](a_node_gid) -> caf::result< result_or_errbox<std::string> > {
				return delegate_source< result_or_errbox<std::string> >(a_node_gid());
			},

			[this](a_lnk_inode) -> caf::result< result_or_errbox<inodeptr> > {
				return delegate_source< result_or_errbox<inodeptr> >(a_lnk_inode());
			},

		}).or_else(super::make_behavior());
	}
};

// impl
sym_link_impl::sym_link_impl(std::string name, std::string path, Flags f)
	: super(std::move(name), f), path_(std::move(path))
{}

sym_link_impl::sym_link_impl()
	: super()
{}

auto sym_link_impl::pointee() const -> result_or_err<link> {
	const auto parent = owner_.lock();
	if(!parent) return tl::make_unexpected( error::quiet(Error::UnboundSymLink) );

	link src_link;
	if(auto er = error::eval_safe([&] { src_link = deref_path(path_, parent); }); er)
		return tl::make_unexpected(std::move(er));
	else if(src_link)
		return src_link;
	return tl::make_unexpected( error::quiet(Error::LinkExpired) );
}

auto sym_link_impl::data() -> result_or_err<sp_obj> {
	auto res = pointee().and_then([](const link& src_link) {
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

auto sym_link_impl::spawn_actor(sp_limpl limpl) const -> caf::actor {
	return spawn_lactor<sym_link_actor>(std::move(limpl));
}

auto sym_link_impl::clone(bool deep) const -> sp_limpl {
	// no deep copy support for symbolic link
	return std::make_shared<sym_link_impl>(name_, path_, flags_);
}

auto sym_link_impl::propagate_handle(const link& super) -> result_or_err<sp_node> {
	// weak link cannot be a node's handle
	return actorf<result_or_errbox<sp_node>>(super, a_lnk_dnode(), true);
}

LIMPL_TYPE_DEF(sym_link_impl, "sym_link")

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

LINK_CONVERT_TO(sym_link)

bool sym_link::check_alive() {
	auto res = bool(deref_path(SIMPL.path_, owner()));
	auto S = res ? ReqStatus::OK : ReqStatus::Error;
	rs_reset_if_neq(Req::Data, S, S);
	return res;
}

/// return stored pointee path
std::string sym_link::src_path(bool human_readable) const {
	if(!human_readable) return SIMPL.path_;
	else if(const auto parent = owner())
		return convert_path(SIMPL.path_, parent->handle());
	return {};
}

auto sym_link::type_id_() -> std::string_view {
	return sym_link_impl::type_id_();
}

NAMESPACE_END(blue_sky::tree)
