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

	auto pointee() const -> result_or_err<sp_link> {
		return static_cast<sym_link_impl&>(impl).pointee();
	}

	//// make data request via source link
	//auto data_ex(obj_processor_f cb, ReqOpts opts) -> void override {
	//	pointee().and_then([&](const sp_link& src_link) {
	//		if(enumval(opts & ReqOpts::DirectInvoke))
	//			cb(src_link->data_ex());
	//		else
	//			caf::anon_send(src_link->actor(), a_lnk_data(), std::move(cb), opts);
	//	});
	//}

	//auto data_node_ex(node_processor_f cb, ReqOpts opts) -> void override {
	//	pointee().and_then([&](const sp_link& src_link) {
	//		if(enumval(opts & ReqOpts::DirectInvoke))
	//			cb(src_link->data_node_ex());
	//		else
	//			caf::anon_send(src_link->actor(), a_lnk_dnode(), std::move(cb), opts);
	//	});
	//}

	template<typename R, typename... Args>
	auto delegate_source(Args&&... args)
	-> std::enable_if_t<tl::detail::is_expected<R>::value, caf::result<R>> {
		if(auto p = pointee())
			return delegate(link::actor(*p.value()), std::forward<Args>(args)...);
		else
			return R{ tl::unexpect, p.error() };
	}

	template<typename R, typename... Args>
	auto delegate_source(R errval, Args&&... args)
	-> std::enable_if_t<!tl::detail::is_expected<R>::value, caf::result<R>> {
		if(auto p = pointee())
			return delegate(link::actor(*p.value()), std::forward<Args>(args)...);
		else
			return errval;
	}

	// delegate name, OID, etc requests to source link
	auto make_behavior() -> caf::behavior override {
		return caf::message_handler({

			[this](a_lnk_oid) -> caf::result<std::string> {
				return delegate_source<std::string>(type_descriptor::nil().name, a_lnk_oid());
			},

			[this](a_lnk_otid) -> caf::result<std::string> {
				return delegate_source<std::string>(type_descriptor::nil().name, a_lnk_otid());
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

auto sym_link_impl::pointee() const -> result_or_err<sp_link> {
	const auto parent = owner_.lock();
	if(!parent) return tl::make_unexpected( error::quiet(Error::UnboundSymLink) );

	sp_link src_link;
	if(auto er = error::eval_safe([&] { src_link = deref_path(path_, parent); }); er)
		return tl::make_unexpected(std::move(er));
	else if(src_link)
		return src_link;
	return tl::make_unexpected( error::quiet(Error::LinkExpired) );
}

auto sym_link_impl::data() -> result_or_err<sp_obj> {
	auto res = pointee().and_then([](const sp_link& src_link) {
		return src_link->data_ex();
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

///////////////////////////////////////////////////////////////////////////////
//  class
//
/// ctor -- pointee is specified by string path
sym_link::sym_link(std::string name, std::string path, Flags f)
	: link(std::make_shared<sym_link_impl>(std::move(name), std::move(path), f))
{}
/// ctor -- pointee is specified directly - absolute path will be stored
sym_link::sym_link(std::string name, const sp_link& src, Flags f)
	: sym_link(std::move(name), abspath(src), f)
{}

sym_link::sym_link()
	: super(std::make_shared<sym_link_impl>(), false)
{}

/// implement link's API
sp_link sym_link::clone(bool deep) const {
	// no deep copy support for symbolic link
	return std::make_shared<sym_link>(pimpl()->name_, pimpl()->path_, flags());
}

std::string sym_link::type_id() const {
	return "sym_link";
}

auto sym_link::pimpl() const -> sym_link_impl* {
	return static_cast<sym_link_impl*>(super::pimpl());
}

bool sym_link::check_alive() {
	auto res = bool(deref_path(pimpl()->path_, owner()));
	auto S = res ? ReqStatus::OK : ReqStatus::Error;
	rs_reset_if_neq(Req::Data, S, S);
	return res;
}

/// return stored pointee path
std::string sym_link::src_path(bool human_readable) const {
	if(!human_readable) return pimpl()->path_;
	else if(const auto parent = owner())
		return convert_path(pimpl()->path_, parent->handle());
	return {};
}

auto sym_link::propagate_handle() -> result_or_err<sp_node> {
	// sym link cannot be a node's handle
	return data_node_ex();
}

NAMESPACE_END(blue_sky::tree)
