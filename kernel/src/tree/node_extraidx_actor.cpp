/// @file
/// @author uentity
/// @date 29.01.2020
/// @brief Node's extra indexes support actor impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_extraidx_actor.h"
#include "node_leafs_storage.h"

#include <bs/kernel/radio.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

#include <caf/typed_event_based_actor.hpp>
//#include <caf/actor_ostream.hpp>
//#include <bs/log.h>

#include <algorithm>
#include <unordered_map>

OMIT_OBJ_SERIALIZATION

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

NAMESPACE_BEGIN()

// returns memoized key extractor
template<Key K>
auto memo_kex() {
	return [kex = Key_tag<K>(), memo = std::unordered_map<const link*, Key_type<K>>{}](
		const link& L
	) mutable {
		if(auto p = memo.find(&L); p != memo.end()) return p->second;
		auto k = kex(L);
		memo[&L] = k;
		return k;
	};
}

// memoized operator <
template<Key K>
auto memo_less() {
	return [kex = memo_kex<K>()](const auto& x1, const auto& x2) mutable {
		return kex(x1) < kex(x2);
	};
}

// search predicate, non-memoized because we do single linear pass
template<Key K>
auto equal_to(const Key_type<K>& rhs) {
	return [&, kex = Key_tag<K>()](const auto& x) {
		return kex(x) == rhs;
	};
}

auto sort(Key order, links_v& leafs) -> void {
	switch(order) {
	default:
	case Key::OID:
		std::sort(leafs.begin(), leafs.end(), memo_less<Key::OID>());
		break;
	case Key::Type:
		std::sort(leafs.begin(), leafs.end(), memo_less<Key::Type>());
		break;
	case Key::Name:
		std::sort(leafs.begin(), leafs.end(), memo_less<Key::Name>());
		break;
	case Key::ID:
		std::sort(leafs.begin(), leafs.end(), memo_less<Key::ID>());
		break;
	}
}

// find among leafs
auto find(const std::string& key, Key meaning, const links_v& leafs) {
	switch(meaning) {
	default:
	case Key::OID:
		return std::find_if(leafs.begin(), leafs.end(), equal_to<Key::OID>(key));
	case Key::Type:
		return std::find_if(leafs.begin(), leafs.end(), equal_to<Key::Type>(key));
	case Key::Name:
		return std::find_if(leafs.begin(), leafs.end(), equal_to<Key::Name>(key));
	}
}

template<Key K>
auto equal_range(const Key_type<K>& key, const links_v& leafs) {
	auto equal_key = equal_to<K>(key);
	auto res = links_v{};
	auto p = leafs.begin(), end = leafs.end();
	while(p != end) {
		if(p = std::find_if(p, end, equal_key); p != end)
			res.push_back(*p++);
	}
	return res;
}

auto equal_range(const std::string& key, Key meaning, const links_v& leafs) {
	switch(meaning) {
	default:
	case Key::Name:
		return equal_range<Key::Name>(key, leafs);
	case Key::OID:
		return equal_range<Key::OID>(key, leafs);
	case Key::Type:
		return equal_range<Key::Type>(key, leafs);
	}
}

// [NOTE] this overload set is a replacement of nice 'constexpr if' for VS2017
// It still tries to check discarded 'if constexpr' path even in templated context
template<Key K>
auto call_find(const caf::scoped_actor& f, const node::actor_type& Nactor, Key_type<K> key)
-> std::enable_if_t<K == Key::ID, link> {
	return actorf<link>(f, Nactor, infinite, a_node_find(), std::move(key)).value_or(link{});
};
template<Key K>
auto call_find(const caf::scoped_actor& f, const node::actor_type& Nactor, Key_type<K> key)
-> std::enable_if_t<K != Key::ID, link> {
	return actorf<link>(f, Nactor, infinite, a_node_find(), std::move(key), K).value_or(link{});
};

// deep search
template<Key K = Key::ID>
auto deep_search(
	const caf::scoped_actor& f, node::actor_type Nactor, const Key_type<K>& key,
	std::set<lid_type> active_symlinks = {}
) -> link {
	// first do direct search in leafs
	if(auto r = call_find<K>(f, Nactor, key)) return r;

	// if not succeeded search in children nodes
	auto leafs = actorf<links_v>(f, Nactor, infinite, a_node_leafs(), Key::AnyOrder).value_or(links_v{});
	for(const auto& l : leafs) {
		// remember symlink
		const auto is_symlink = l.type_id() == sym_link::type_id_();
		if(is_symlink){
			if(active_symlinks.find(l.id()) == active_symlinks.end())
				active_symlinks.insert(l.id());
			else continue;
		}
		// check populated status before moving to next level
		if(l.flags() & LazyLoad && l.req_status(Req::DataNode) != ReqStatus::OK)
			continue;
		// search on next level
		if(auto next_n = l.data_node()) {
			auto next_l = deep_search<K>(f, next_n->actor(), key, active_symlinks);
			if(next_l) return next_l;
		}
		// remove symlink
		if(is_symlink)
			active_symlinks.erase(l.id());
	}
	return {};
}

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  actor behavior
 *-----------------------------------------------------------------------------*/
auto extraidx_search_actor(extraidx_search_api::pointer) -> extraidx_search_api::behavior_type {
return {
	// return leafs sorted by keys K
	[](a_node_leafs, Key order, links_v leafs) -> links_v {
		sort(order, leafs);
		return leafs;
	},

	// find link with given key (link ID index isn't supported)
	[](a_node_find, const std::string& key, Key meaning, const links_v& leafs) -> link {
		if(auto p = find(key, meaning, leafs); p != leafs.end())
			return *p;
		return {};
	},

	// [NOTE] assume leafs are in AnyOrder!
	[](a_node_index, const std::string& key, Key meaning, const links_v& leafs) -> node::existing_index {
		if(auto p = find(key, meaning, leafs); p != leafs.end())
			return p - leafs.begin();
		return {};
	},

	[](a_node_equal_range, const std::string& key, Key meaning, const links_v& leafs) -> links_v {
		return equal_range(key, meaning, leafs);
	},
}; }

auto extraidx_deep_search_actor(extraidx_deep_search_api::pointer self, node_impl::actor_type Nactor)
-> extraidx_deep_search_api::behavior_type { return {
	// deep search
	[=](a_node_deep_search, const lid_type& key) -> link {
		//caf::aout(self) << "==> a_node_deep_search extra" << std::endl;
		return deep_search<Key::ID>(caf::scoped_actor{system()}, Nactor, key);
	},

	[=](a_node_deep_search, const std::string& key, Key meaning) -> link {
		auto f = caf::scoped_actor{system()};
		switch(meaning) {
		default:
		case Key::Name:
			return deep_search<Key::Name>(f, Nactor, key);
		case Key::OID:
			return deep_search<Key::OID>(f, Nactor, key);
		case Key::Type:
			return deep_search<Key::Type>(f, Nactor, key);
		}
	}
}; }

auto extraidx_erase_actor(extraidx_erase_api::pointer self, node_impl::actor_type Nactor)
-> extraidx_erase_api::behavior_type { return {
	[=](a_node_erase, const std::string& key, Key meaning, const links_v& leafs) -> caf::result<std::size_t> {
		auto victims = equal_range(key, meaning, leafs);
		return self->delegate(
			Nactor, a_node_erase(), range_t(victims.begin(), victims.end()).extract_keys()
		);
	}
}; }

NAMESPACE_END(blue_sky::tree)
