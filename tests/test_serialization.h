/// @file
/// @author uentity
/// @date 11.04.2019
/// @brief Helpers for quick test serialization of any type
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/objbase.h>
#include <bs/log.h>
#include <bs/serialize/serialize.h>

#include <boost/interprocess/streams/vectorstream.hpp>
#include <boost/test/unit_test.hpp>

template<bool Binary>
struct select_archives {
	using InputArchive = cereal::JSONInputArchive;
	using OutputArchive = cereal::JSONOutputArchive;
	using Stream = std::stringstream;
};
template<>
struct select_archives<true> {
	using InputArchive = cereal::PortableBinaryInputArchive;
	using OutputArchive = cereal::PortableBinaryOutputArchive;
	using Stream = boost::interprocess::basic_vectorstream< std::vector<char> >;
};

template<bool Binary = false, typename T = blue_sky::sp_obj>
auto test_saveload(const T& obj, bool dump_serialized = true) {
	using namespace blue_sky;
	using namespace blue_sky::log;

	using InputArchive = typename select_archives<Binary>::InputArchive;
	using OutputArchive = typename select_archives<Binary>::OutputArchive;
	using Stream = typename select_archives<Binary>::Stream;

	std::string dump;
	Stream S;
	// dump object into string
	{
		OutputArchive ja(S);
		ja(obj);
		if constexpr(!Binary) {
			if(dump_serialized) {
				dump = S.str();
				bsout() << I("-- Serialized dump:\n{}", dump) << end;
			}
		}
		else { (void)dump_serialized; }
	}
	// load object from dump
	T obj1;
	{
		InputArchive ja(S);
		ja(obj1);
	}

	if constexpr(std::is_convertible_v<T, sp_cobj>)
		BOOST_TEST(obj->id() == obj1->id());
	return obj1;
}

template<typename T>
auto test_json(const T& obj, bool dump_serialized = true) {
	return test_saveload(obj, dump_serialized);
}

template<typename T>
auto test_binary(const T& obj) {
	return test_saveload<true>(obj);
}

