/// @author Alexander Gagarin (@uentity)
/// @date 17.02.2021
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "map_engine.h"

NAMESPACE_BEGIN(blue_sky::tree)

map_impl_base::map_impl_base(bool is_link_mapper) :
	in_(node::nil()), out_(node::nil()), tag_(), update_on_(Event::None), opts_(TreeOpts::Normal),
	is_link_mapper(is_link_mapper)
{}

map_impl_base::map_impl_base(
	bool is_link_mapper_, uuid tag, std::string name,
	const link_or_node& input, const link_or_node& output,
	Event update_on, TreeOpts opts, Flags f
) : super(std::move(name), f | Flags::LazyLoad), tag_(tag), update_on_(update_on), opts_(opts),
	is_link_mapper(is_link_mapper_)
{
	static const auto extract_node = [](node& lhs, auto& rhs) {
		visit(meta::overloaded{
			[&](const link& L) { lhs = L ? L.data_node() : node::nil(); },
			[&](const node& N) { lhs = N; }
		}, rhs);
	};

	extract_node(in_, input);
	extract_node(out_, output);
	// ensure we have valid output node
	if(!out_ || in_ == out_) out_ = node();

	// set initial status values
	rs_reset_quiet(Req::Data, ReqReset::Always, ReqStatus::Error);
	rs_reset_quiet(Req::DataNode, ReqReset::Always, ReqStatus::Void);
}

auto map_impl_base::spawn_actor(sp_limpl Limpl) const -> caf::actor {
	return spawn_lactor<map_link_actor>(std::move(Limpl));
}

auto map_impl_base::propagate_handle() -> node_or_err {
	// if output node doesn't have handle (for ex new node created in ctor) - set it to self
	// [NOTE] check if `out_` is not nil for deserialization case
	if(out_ && !out_.handle())
		super::propagate_handle(out_);
	return out_;
}

// Data request always return error
auto map_impl_base::data() -> obj_or_err { return unexpected_err_quiet(Error::EmptyData); }
auto map_impl_base::data(unsafe_t) const -> sp_obj { return nullptr; }

auto map_impl_base::data_node(unsafe_t) const -> node { return out_; }

NAMESPACE_END(blue_sky::tree)
