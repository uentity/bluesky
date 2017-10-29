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
#include "type_macro.h"

#include <unordered_map>

namespace blue_sky {

typedef std::shared_ptr< objbase > bs_type_ctor_result;
typedef const std::shared_ptr< objbase >& bs_type_copy_param;

typedef bs_type_ctor_result (*BS_TYPE_COPY_FUN)(bs_type_copy_param);
typedef const blue_sky::type_descriptor& (*BS_GET_TD_FUN)();

namespace detail {
/// Convert lambda::operator() to bs type construct function pointer
template <typename L> struct lam2bsctor {};
template <typename C, typename R, typename... A>
struct lam2bsctor<R (C::*)(A...)> { typedef bs_type_ctor_result type(A...); };
template <typename C, typename R, typename... A>
struct lam2bsctor<R (C::*)(A...) const> { typedef bs_type_ctor_result type(A...); };
}

/*!
\struct type_descriptor
\ingroup blue_sky
\brief BlueSky type descriptor
*/
class BS_API type_descriptor {
private:
	friend class kernel;

	const BS_TYPE_INFO bs_ti_;
	const BS_GET_TD_FUN parent_td_fun_;
	mutable BS_TYPE_COPY_FUN copy_fun_;

	// map type of params tuple -> typeless creation function
	using ctor_handle = std::pair< void(*)(), void*>;
	mutable std::unordered_map< std::type_index, ctor_handle > creators_;

	// `decay_helper` extends `std::decay` such that pointed type (for pointers) is also decayed
	template<typename T, typename = void>
	struct decay_helper {
		using type = std::decay_t<T>; //-> can give a pointer, that's why 2nd pass is needed
	};
	template<typename T>
	struct decay_helper<T, std::enable_if_t<std::is_pointer<T>::value>> {
		using type = std::decay_t<decltype(*std::declval<T>())>*;
	};
	// need to double-pass through `decay_helper` to strip `const` from inline strings
	// and other const arrays
	template<typename T> using decay_arg = typename decay_helper< typename decay_helper<T>::type >::type;
	// decayed ctor parameters tuple is the key to find corresponding creator function
	template< typename... Args > using args_pack = std::tuple<decay_arg<Args>...>;

	// generate BS type creator signature with consistent results
	// if param is pointer (const or not) to type `T` -> result is `const T*`
	// else result is const reference to decayed param type `T`
	template<typename T, typename = void>
	struct pass_helper {
		using type = std::add_lvalue_reference_t< std::add_const_t<decay_arg<T>> >;
	};
	template<typename T>
	struct pass_helper<T, std::enable_if_t<std::is_pointer<decay_arg<T>>::value>> {
		using dT = std::remove_reference_t<decltype(*std::declval<decay_arg<T>>())>;
		using type = std::add_const_t<dT>*;
	};
	template<typename T> using pass_arg = typename pass_helper<T>::type;
	// final signature of object creator function callback
	template< typename... Args >
	using creator_callback = bs_type_ctor_result (*)(void*, pass_arg<Args>...);

	template< class T, class unused = void >
	struct extract_tdfun {
		static BS_GET_TD_FUN go() {
			return T::bs_type;
		}
	};
	template < class unused >
	struct extract_tdfun< nil, unused > {
		static BS_GET_TD_FUN go() {
			return nullptr;
		}
	};

	template< class T>
	struct extract_typename {
		template< class str_type >
		static std::string go(str_type val) {
			return val;
		}

		static std::string go(std::nullptr_t) {
			return bs_type_name< T >();
		}
	};

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
		template< typename T >
		operator std::shared_ptr< T >() const && {
			return std::static_pointer_cast< T, objbase >(ptr_);
		}

		bs_type_ctor_result ptr_;
	};

	// should we add default ctor?
	template< typename T, bool Enable >
	void add_def_constructor(std::enable_if_t< !Enable >* = nullptr) {}
	template< typename T, bool Enable >
	void add_def_constructor(std::enable_if_t< Enable >* = nullptr) {
		add_constructor< T >();
	}

	// should we add default copy?
	template< typename T, bool Enable >
	void add_def_copy_constructor(std::enable_if_t< !Enable >* = nullptr) {}
	template< typename T, bool Enable >
	void add_def_copy_constructor(std::enable_if_t< Enable >* = nullptr) {
		add_copy_constructor< T >();
	}

public:
	const std::string name; //!< string type name
	const std::string description; //!< arbitrary type description

	// default constructor - type_descriptor points to nil
	type_descriptor() :
		bs_ti_(nil_type_info()), parent_td_fun_(nullptr), copy_fun_(nullptr),
		name(nil().name)
	{}

	// constructor from string type name for temporary tasks (searching etc)
	type_descriptor(const char* type_name) :
		bs_ti_(nil_type_info()), parent_td_fun_(nullptr), copy_fun_(nullptr),
		name(type_name)
	{}

	// constructor from BS_TYPE_INFO
	type_descriptor(const BS_TYPE_INFO& ti) :
		bs_ti_(ti), parent_td_fun_(nullptr), copy_fun_(nullptr),
		name(nil().name)
	{}

	// standard constructor
	type_descriptor(
		const BS_TYPE_INFO& ti, const char* type_name, const BS_TYPE_COPY_FUN& cp_fn,
		const BS_GET_TD_FUN& parent_td_fn, const char* description = ""
	) :
		bs_ti_(ti), parent_td_fun_(parent_td_fn), copy_fun_(cp_fn),
		name(type_name), description(description)
	{}

	// templated ctor for BlueSky types
	// if add_def_construct is set -- add default (empty) type's constructor
	// if add_def_copy is set -- add copy constructor
	template<
		class T, class base = nil, class typename_t = std::nullptr_t,
		bool add_def_ctor = false, bool add_def_copy = false
	>
	type_descriptor(
		identity< T >, identity< base >,
		typename_t type_name = nullptr, const char* description = nullptr,
		std::integral_constant< bool, add_def_ctor > = std::false_type(),
		std::integral_constant< bool, add_def_copy > = std::false_type()
	) :
		bs_ti_(BS_GET_TI(T)), parent_td_fun_(extract_tdfun< base >::go()), copy_fun_(nullptr),
		name(extract_typename< T >::go(type_name)), description(description)
	{
		add_def_constructor< T, add_def_ctor >();
		add_def_copy_constructor< T, add_def_copy >();
	}

	// obtain Nil type_descriptor
	static const type_descriptor& nil();

	/*-----------------------------------------------------------------
	 * create new instance
	 *----------------------------------------------------------------*/
	// vanilla fucntion pointer as type constructor
	// NOTE: if Args&& used as function params then args types auto-deduction fails
	// for templated functions
	template< typename... Args >
	void add_constructor(bs_type_ctor_result (*f)(Args...)) const {
		creators_[typeid(args_pack< Args... >)] = ctor_handle(
			reinterpret_cast< void(*)() >((creator_callback< Args... >)
				[](void* ff, pass_arg< Args >... args) {
					return (*reinterpret_cast< decltype(f) >(ff))(args...);
				}
			),
			reinterpret_cast< void* >(f)
		);
	}

	// add stateless lambda as type constructor
	template< typename Lambda >
	void add_constructor(Lambda&& f) const {
		using func_t = typename detail::lam2bsctor<decltype(&std::remove_reference<Lambda>::type::operator())>::type;
		add_constructor((func_t*)f);
	}

	// std type construction
	// explicit constructor arguments types
	template< typename T, typename... Args >
	void add_constructor() const {
		creators_[typeid(args_pack< Args... >)] = ctor_handle(
			reinterpret_cast< void(*)() >((creator_callback< Args... >)
				[](void* ff, pass_arg< Args >... args) {
					return std::static_pointer_cast<objbase, T>(std::make_shared<T>(args...));
				}
			),
			nullptr
		);
	}

	// std type's constructor
	// deduce constructor arguments types from passed tuple
	template< typename T, typename... Args >
	void add_constructor(std::tuple< Args... >*) const {
		add_constructor< T, Args... >();
	}

	// make new instance
	template< typename... Args >
	shared_ptr_cast construct(Args&&... args) const {
		auto creator = creators_.find(typeid(args_pack< Args... >));
		if(creator != creators_.end()) {
			auto callback = reinterpret_cast< creator_callback<Args...> >(creator->second.first);
			return callback(creator->second.second, std::forward<Args>(args)...);
		}
		return {};
	}

	/*-----------------------------------------------------------------
	 * create copy of instance
	 *----------------------------------------------------------------*/
	// register type's copy constructor
	template< typename T >
	void add_copy_constructor() const {
		copy_fun_ = [](bs_type_copy_param src) {
			return std::static_pointer_cast< objbase, T >(
				std::make_shared< T >(static_cast< const T& >(*src))
			);
		};
	}

	// construct copy using vanilla function
	void add_copy_constructor(BS_TYPE_COPY_FUN f) const {
		copy_fun_ = f;
	}

	// make instance copy
	shared_ptr_cast clone(bs_type_copy_param src) const {
		if(copy_fun_)
			return (*copy_fun_)(src);
		return {};
	}

	/// read access to base fields
	BS_TYPE_INFO type() const {
		return bs_ti_;
	};

	/// tests
	bool is_nil() const {
		return ::blue_sky::is_nil(bs_ti_);
	}
	bool is_copyable() const {
		return (copy_fun_ != nullptr);
	}

	/// conversions
	operator std::string() const {
		return name;
	}
	operator const char*() const {
		return name.c_str();
	}

	//! by default type_descriptors are comparable by bs_type_info
	bool operator <(const type_descriptor& td) const {
		return bs_ti_ < td.bs_ti_;
	}

	//! retrieve type_descriptor of parent class
	const type_descriptor& parent_td() const {
		if(parent_td_fun_)
			return (*parent_td_fun_)();
		else
			return nil();
	}
};

// comparison with type string
inline bool operator <(const type_descriptor& td, const std::string& type_string) {
	return (td.name < type_string);
}

inline bool operator ==(const type_descriptor& td, const std::string& type_string) {
	return (td.name == type_string);
}

inline bool operator !=(const type_descriptor& td, const std::string& type_string) {
	return td.name != type_string;
}

// comparison with bs_type_info
inline bool operator <(const type_descriptor& td, const BS_TYPE_INFO& ti) {
	return (td.type() < ti);
}

inline bool operator ==(const type_descriptor& td, const BS_TYPE_INFO& ti) {
	return (td.type() == ti);
}

inline bool operator !=(const type_descriptor& td, const BS_TYPE_INFO& ti) {
	return !(td.type() == ti);
}

// upcastable_eq(td1, td2) will return true if td1 != td2
// but td1 can be casted up to td2 (i.e. td1 is inherited from td1)
struct BS_API upcastable_eq : public std::binary_function<
	type_descriptor, type_descriptor, bool
> {
	bool operator()(const type_descriptor& td1, const type_descriptor& td2) const;
};

}	// eof blue_sky namespace

