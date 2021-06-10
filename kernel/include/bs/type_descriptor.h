/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Type descriptor class for BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "error.h"
#include "type_assign.h"
#include "type_macro.h"
#include "propdict.h"
#include "detail/function_view.h"

#include <unordered_map>

NAMESPACE_BEGIN(blue_sky)

using bs_type_ctor_result = std::shared_ptr<objbase>;
using bs_type_copy_param = const std::shared_ptr<const objbase>&;

using BS_TYPE_COPY_FUN = bs_type_ctor_result (*)(bs_type_copy_param);
using BS_GET_TD_FUN = const blue_sky::type_descriptor& (*)();

/*-----------------------------------------------------------------------------
 *  describes BS type and provides create/copy/assign facilities
 *-----------------------------------------------------------------------------*/
class BS_API type_descriptor {
private:
	const BS_GET_TD_FUN parent_td_fun_;
	const function_view<detail::deduce_callable_t<BS_TYPE_ASSIGN_FUN>> assign_fun_;
	mutable BS_TYPE_COPY_FUN copy_fun_;

	// map type of params tuple -> typeless creation function
	using erased_ctor = function_view<void()>;
	using ctor_handle = std::pair<erased_ctor, erased_ctor>;
	mutable std::unordered_map<std::type_index, ctor_handle> creators_;

	// `decay_helper` extends `std::decay` such that pointed type (for pointers) is also decayed
	template<typename T, typename = void>
	struct decay_helper {
		using type = std::decay_t<T>; //-> can give a pointer, that's why 2nd pass is needed
	};
	template<typename T>
	struct decay_helper<T, std::enable_if_t<std::is_pointer_v<T>>> {
		using type = std::decay_t<decltype(*std::declval<T>())>*;
	};
	// need to double-pass through `decay_helper` to strip `const` from inline strings
	// and other const arrays
	template<typename T> using decay_arg = typename decay_helper< typename decay_helper<T>::type >::type;

	// generate BS type creator signature with consistent results
	// if param is pointer (const or not) to type `T` -> result is `const T*`
	// else result is const reference to decayed param type `T`
	template<typename T, typename = void>
	struct pass_helper {
		using type = std::add_lvalue_reference_t< std::add_const_t<decay_arg<T>> >;
	};
	template<typename T>
	struct pass_helper<T, std::enable_if_t<std::is_pointer_v<decay_arg<T>>>> {
		using dT = std::remove_reference_t<decltype(*std::declval<decay_arg<T>>())>;
		using type = std::add_const_t<dT>*;
	};
	template<typename T> using pass_arg = typename pass_helper<T>::type;

	// normalized signature of object constructor
	template<typename... Args>
	struct normalized_ctor {
		using type = function_view<bs_type_ctor_result (erased_ctor&, pass_arg<Args>...)>;

		template<typename F>
		static auto make() {
			return type{[](erased_ctor& ec, auto&&... args) {
				return (reinterpret_cast<F&>(ec))(std::forward<decltype(args)>(args)...);
			}};
		}
	};

	template<typename R, typename... Args>
	struct normalized_ctor<R (Args...)> : normalized_ctor<Args...> {};

	template<typename... Args> using normalized_ctor_t = typename normalized_ctor<Args...>::type;

public:
	/// string type name
	const std::string name;
	/// arbitrary type description
	const std::string description;

	// std::shared_ptr support casting only using explicit call to std::static_pointer_cast
	// this helper allows to avoid lengthy typing and auto-cast pointer from objbase
	// to target type
	struct shared_ptr_cast {
		shared_ptr_cast() : ptr_(nullptr) {}
		// only move semantics allowed
		shared_ptr_cast(const shared_ptr_cast&) = delete;
		shared_ptr_cast& operator=(const shared_ptr_cast&) = delete;
		shared_ptr_cast(shared_ptr_cast&&) = default;
		// accept only rvalues and move them to omit ref incrementing
		shared_ptr_cast(std::shared_ptr< objbase>&& rhs) : ptr_(std::move(rhs)) {}

		// only allow imlicit conversion of rvalue shared_ptr_cast (&&)
		template<typename T>
		operator std::shared_ptr<T>() const && {
			return std::static_pointer_cast<T>(std::move(ptr_));
		}

		bs_type_ctor_result ptr_;
	};

	// constructor from string type name for temporary tasks (searching etc)
	type_descriptor(std::string_view type_name = {});

	// standard constructor
	type_descriptor(
		std::string type_name, BS_GET_TD_FUN parent_td_fn, BS_TYPE_ASSIGN_FUN assign_fn,
		BS_TYPE_COPY_FUN cp_fn = nullptr, std::string description = {}
	);

	// templated ctor for BlueSky types
	template<class T, class base, class typename_t = std::nullptr_t>
	type_descriptor(
		identity< T >, identity< base >,
		typename_t type_name = nullptr, std::string description = {}
	) :
		parent_td_fun_(&base::bs_type),
		assign_fun_(blue_sky::detail::make_assigner<T>()),
		copy_fun_(nullptr),
		name([&] {
			if constexpr(std::is_same_v<typename_t, std::nullptr_t>)
				return bs_type_name<T>();
			else
				return std::move(type_name);
		}()),
		description(std::move(description))
	{
		// auto-add default ctor, from single string arg (typically ID), and copy ctor
		if constexpr(std::is_default_constructible_v<T>)
			add_constructor<T>();
		if constexpr(std::is_constructible_v<T, std::string>)
			add_constructor<T, std::string>();
		if constexpr(std::is_copy_constructible_v<T>)
			add_copy_constructor<T>();
	}

	///////////////////////////////////////////////////////////////////////////////
	// register constructors & construct new instance from ctor agruments 
	//
	template<typename F>
	auto add_constructor(F&& f) const -> void {
		using Finfo = deduce_callable<F>;
		using Fsign = typename Finfo::type;
		static_assert(
			std::is_same_v<typename Finfo::result, bs_type_ctor_result>,
			"Type constructor must return sp_obj"
		);
		static_assert(
			std::is_convertible_v<F, std::add_pointer_t<Fsign>>,
			"Only stateless functor that can be registered as constructor"
		);

		// store type-erased constructor
		using ctor_t = normalized_ctor<Fsign>;
		auto src_f = function_view{std::forward<F>(f)};
		auto creator_f = ctor_t::template make<decltype(src_f)>();

		creators_.insert_or_assign(typeid(typename ctor_t::type), ctor_handle{
			reinterpret_cast<erased_ctor&>(creator_f), reinterpret_cast<erased_ctor&>(src_f)
		});
	}

	template<typename T, typename... Args>
	auto add_constructor() const -> void {
		add_constructor([](pass_arg<Args>... args) {
			return std::static_pointer_cast<objbase>(std::make_shared<T>(args...));
		});
	}

	// make new instance
	template<typename... Args>
	auto construct(Args&&... args) const -> shared_ptr_cast {
		using Fnorm = normalized_ctor_t<Args...>;
		if(auto creator = creators_.find(typeid(Fnorm)); creator != creators_.end()) {
			auto& [creator_f, src_f] = creator->second;
			return (reinterpret_cast<Fnorm&>(creator_f))(src_f, std::forward<Args>(args)...);
		}
		return {};
	}

	///////////////////////////////////////////////////////////////////////////////
	// make copy of instance 
	//
	// register type's copy constructor
	template< typename T >
	auto add_copy_constructor() const -> void {
		copy_fun_ = [](bs_type_copy_param src) {
			return src ? std::static_pointer_cast< objbase, T >(
				std::make_shared< T >(static_cast< const T& >(*src))
			) : nullptr;
		};
	}

	// construct copy using vanilla function
	auto add_copy_constructor(BS_TYPE_COPY_FUN f) const -> void {
		copy_fun_ = f;
	}

	// make a copy of object instance
	auto clone(bs_type_copy_param src) const -> shared_ptr_cast;

	// assign content from source -> target
	auto assign(sp_obj target, sp_obj source, prop::propdict params = {}) const -> error;

	auto is_copyable() const -> bool;

	static auto nil() -> const type_descriptor&;
	auto is_nil() const -> bool;

	operator std::string() const {
		return name;
	}
	operator const char*() const {
		return name.c_str();
	}

	/// check that object's type descriptor mathes self
	auto isinstance(const sp_cobj& obj) const -> bool;
	
	/// retrieve type_descriptor of parent class
	auto parent_td() const -> const type_descriptor&;

	/// type_descriptors are comparable by string type name
	auto operator <(const type_descriptor& td) const -> bool;

	/// comparison with string type ID
	friend auto operator ==(const type_descriptor& td, std::string_view type_id) -> bool {
		return (td.name == type_id);
	}
	friend auto operator ==(std::string_view type_id, const type_descriptor& td) -> bool {
		return (td.name == type_id);
	}

	friend auto operator !=(const type_descriptor& td, std::string_view type_id) -> bool {
		return td.name != type_id;
	}
	friend auto operator !=(std::string_view type_id, const type_descriptor& td) -> bool {
		return td.name != type_id;
	}

	friend auto operator <(const type_descriptor& td, std::string_view type_id) -> bool {
		return td.name < type_id;
	}
};

inline auto isinstance(const sp_cobj& obj, const type_descriptor& td) -> bool { return td.isinstance(obj); }
BS_API auto isinstance(const sp_cobj& obj, std::string_view obj_type_id) -> bool;

// upcastable_eq(td1, td2) will return true if td1 == td2
// or td1 can be casted up to td2 (i.e. td2 is inherited from td1)
struct BS_API upcastable_eq {
	bool operator()(const type_descriptor& td1, const type_descriptor& td2) const;
};

NAMESPACE_END(blue_sky)

