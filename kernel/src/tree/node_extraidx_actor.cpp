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

#include <caf/typed_event_based_actor.hpp>
//#include <caf/actor_ostream.hpp>
//#include <bs/log.h>

#include <algorithm>
#include <iterator>
#include <unordered_map>

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

NAMESPACE_BEGIN()

// returns memoized key extractor
template<Key K>
auto memo_kex() {
	return [kex = Key_tag<K>(), memo = std::unordered_map<const link*, Key_type<K>>{}](
		const link& L
	) mutable -> decltype(auto) {
		if(auto p = memo.find(&L); p != memo.end()) return p->second;
		return memo.emplace(&L, kex(L)).first->second;
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
	default:
		break;
	}
}

// find among leafs
auto find(const std::string& key, Key meaning, const links_v& leafs) {
	switch(meaning) {
	case Key::OID:
		return std::find_if(leafs.begin(), leafs.end(), equal_to<Key::OID>(key));
	case Key::Type:
		return std::find_if(leafs.begin(), leafs.end(), equal_to<Key::Type>(key));
	case Key::Name:
		return std::find_if(leafs.begin(), leafs.end(), equal_to<Key::Name>(key));
	default:
		return leafs.end();
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
	case Key::Name:
		return equal_range<Key::Name>(key, leafs);
	case Key::OID:
		return equal_range<Key::OID>(key, leafs);
	case Key::Type:
		return equal_range<Key::Type>(key, leafs);
	default:
		return links_v{};
	}
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

	// extract keys
	[](a_node_keys, Key order, links_v leafs) -> lids_v {
		sort(order, leafs);
		return node_impl::keys<Key::ID>(leafs);
	},

	[](a_node_keys, Key meaning, Key order, links_v leafs) -> std::vector<std::string> {
		sort(order, leafs);

		switch(meaning) {
		case Key::ID :
			return range_t{leafs.begin(), leafs.end()}.extract<std::string>(
				[](const auto& L) { return to_string(L.id()); }
			);
		case Key::Name:
			return node_impl::keys<Key::Name>(leafs);
		case Key::OID:
			return node_impl::keys<Key::OID>(leafs);
		case Key::Type:
			return node_impl::keys<Key::Type>(leafs);
		default:
			return {};
		}
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
