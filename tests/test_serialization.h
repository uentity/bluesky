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
	using StreamData = std::string;

	static auto stream_data(const Stream& S) -> StreamData { return S.str(); }
};
template<>
struct select_archives<true> {
	using InputArchive = cereal::PortableBinaryInputArchive;
	using OutputArchive = cereal::PortableBinaryOutputArchive;
	using Stream = boost::interprocess::basic_vectorstream< std::vector<char> >;
	using StreamData = std::vector<char>;

	static auto stream_data(const Stream& S) -> StreamData { return S.vector(); }
};


template<bool Binary = false, typename T = blue_sky::sp_obj>
auto test_save(const T& obj, bool dump_serialized = true) {
	using namespace blue_sky;
	using namespace blue_sky::log;

	using Traits = select_archives<Binary>;
	using OutputArchive = typename Traits::OutputArchive;
	using Stream = typename Traits::Stream;

	std::string dump;
	Stream S;
	{
		OutputArchive ja(S);
		ja(obj);
		ja.serializeDeferments();
		if constexpr(!Binary) {
			if(dump_serialized) {
				dump = S.str();
				bsout() << I("-- Serialized dump:\n{}", dump) << end;
			}
		}
		else { (void)dump_serialized; }
	}
	return Traits::stream_data(S);
}

template<bool Binary = false, typename T = blue_sky::sp_obj>
auto test_load(typename select_archives<Binary>::StreamData data, T& obj) -> void {
	using namespace blue_sky;
	using Traits = select_archives<Binary>;
	using InputArchive = typename Traits::InputArchive;
	using Stream = typename Traits::Stream;

	Stream S(std::move(data));
	{
		InputArchive ja(S);
		ja(obj);
		ja.serializeDeferments();
	}
}

template<bool Binary = false, typename T = blue_sky::sp_obj>
auto test_saveload(const T& obj, bool dump_serialized = true) {
	T obj1;
	test_load<Binary>(test_save<Binary>(obj, dump_serialized), obj1);
	if constexpr(std::is_convertible_v<T, blue_sky::sp_cobj>)
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

