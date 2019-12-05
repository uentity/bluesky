/// @file
/// @author uentity
/// @date 22.11.2019
/// @brief Implements data request
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "request_impl.h"

OMIT_OBJ_SERIALIZATION

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

NAMESPACE_BEGIN()

template<bool ManageStatus = true, typename C>
auto data_node_request(link_actor& LA, ReqOpts opts, C&& res_processor) {
	request_impl<ManageStatus>(
		LA, Req::DataNode, opts,
		[&LA, opts]() mutable -> result_or_errbox<sp_node> {
			// directly invoke 'Data' request, store returned value in `res` and return it
			auto res = result_or_errbox<sp_node>{};
			request_impl<ManageStatus>(
				LA, Req::Data, opts | ReqOpts::DirectInvoke,
				[&LA] {
					return LA.pimpl_->data().and_then([](sp_obj&& obj) {
						return obj && obj->is_node() ?
							result_or_err<sp_node>(std::static_pointer_cast<tree::node>(std::move(obj))) :
							tl::make_unexpected(error::quiet(Error::NotANode));
					});
				},
				[&res](result_or_errbox<sp_node>&& N) { res = std::move(N); }
			);
			return res;
		},
		std::forward<C>(res_processor)
	);
}

NAMESPACE_END()

auto link_actor::data_ex(obj_processor_f cb, ReqOpts opts) -> void {
	request_impl(
		*this, Req::Data, opts,
		[Limpl = pimpl_] { return Limpl->data(); },
		std::move(cb)
	);
}

auto link_actor::data_node_ex(node_processor_f cb, ReqOpts opts) -> void {
	data_node_request(*this, opts, std::move(cb));
}

auto cached_link_actor::data_ex(obj_processor_f cb, ReqOpts opts) -> void {
	request_impl(
		*this, Req::Data, opts | ReqOpts::HasDataCache,
		[Limpl = pimpl_] { return Limpl->data(); },
		std::move(cb)
	);
}

auto cached_link_actor::data_node_ex(node_processor_f cb, ReqOpts opts) -> void {
	data_node_request(*this, opts | ReqOpts::HasDataCache, std::move(cb));
}

auto fast_link_actor::data_ex(obj_processor_f cb, ReqOpts opts) -> void {
	// directly invoke `impl.data()` regardless of any options
	request_impl<false>(
		*this, Req::Data, opts | ReqOpts::DirectInvoke,
		[Limpl = pimpl_] { return Limpl->data(); },
		std::move(cb)
	);
}

auto fast_link_actor::data_node_ex(node_processor_f cb, ReqOpts opts) -> void {
	data_node_request<false>(*this, opts | ReqOpts::DirectInvoke, std::move(cb));
}

NAMESPACE_END(blue_sky::tree)
