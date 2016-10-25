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
#include "type_info.h"
//#include "type_macro.h"

#include <unordered_map>
#include <set>
#include <list>
#include <boost/preprocessor/seq/for_each.hpp>

#define BS_NIL_TYPE_TAG "__blue_sky_nil_type__"

namespace blue_sky {

typedef std::shared_ptr< objbase > bs_type_ctor_result;
typedef const std::shared_ptr< objbase >& bs_type_copy_param;
//typedef objbase* (*BS_TYPE_CREATION_FUN)(const std::shared_ptr< objbase >&);
typedef bs_type_ctor_result (*BS_TYPE_COPY_FUN)(bs_type_copy_param);
typedef const blue_sky::type_descriptor& (*BS_GET_TD_FUN)();

typedef std::set< std::shared_ptr< objbase > > bs_objinst_holder;
typedef std::shared_ptr< bs_objinst_holder > sp_objinst;

template< class BST > struct gen_type_descriptor;

/*!
\struct type_descriptor
\ingroup blue_sky
\brief BlueSky type descriptor
*/
class BS_API type_descriptor {
private:
	friend class kernel;
	static unsigned int self_version();

	BS_TYPE_INFO bs_ti_;
	std::string type_name_; //!< string type name
	std::string description_; //!< Short description of type

	mutable BS_GET_TD_FUN parent_td_fun_;
	mutable BS_TYPE_COPY_FUN copy_fun_;

	//BS_TYPE_CREATION_FUN creation_fun_;
	// map type of params tuple -> typeless creation function
	mutable std::unordered_map< std::type_index, std::pair< void(*)(), void*> > creators_;

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
		shared_ptr_cast(shared_ptr_cast&&) = default;
		// accept only rvalues and move them to omit ref incrementing
		shared_ptr_cast(std::shared_ptr< objbase>&& rhs) : ptr_(std::move(rhs)) {}

		template< typename T >
		operator std::shared_ptr< T >() const {
			return std::static_pointer_cast< T, objbase >(ptr_);
		}

		bs_type_ctor_result ptr_;
	};

	// helper to find out parameters pack tuple type
	template< typename... Args >
	using args_pack = std::tuple< std::decay_t< Args >... >;

	// type of object construction callback
	template< typename... Args >
	using creator_callback = bs_type_ctor_result (*)(
		void*, std::add_lvalue_reference_t< const std::decay_t< Args > >...
	);

public:
	// default constructor - type_descriptor points to nil
	type_descriptor() :
		bs_ti_(nil_type_info()), type_name_(BS_NIL_TYPE_TAG),
		parent_td_fun_(nullptr), copy_fun_(nullptr)
	{}

	// Nil constructor for temporary tasks (searching etc)
	type_descriptor(const std::string& type_name) :
		bs_ti_(nil_type_info()), type_name_(type_name), parent_td_fun_(nullptr), copy_fun_(nullptr)
	{}

	// standard constructor
	type_descriptor(
		const BS_TYPE_INFO& ti, const char* type_name, const BS_TYPE_COPY_FUN& cp_fn,
		const BS_GET_TD_FUN& parent_td_fn, const char* description = ""
	) :
		bs_ti_(ti), type_name_(type_name), description_(description),
		parent_td_fun_(parent_td_fn), copy_fun_(cp_fn)
	{}

	// templated ctor for BlueSky types
	// if add_def_construct is set -- add default (empty) type's constructor
	// if add_def_copy is set -- add copy constructor
	template< class T, class base = nil, class typename_t = std::nullptr_t >
	type_descriptor(
		identity< T >, identity< base >,
		typename_t type_name = nullptr, const char* description = nullptr,
		bool add_def_construct = false, bool add_def_copy = false
	) :
		bs_ti_(BS_GET_TI(T)), type_name_(extract_typename< T >::go(type_name)), description_(description),
		parent_td_fun_(extract_tdfun< base >::go()), copy_fun_(nullptr)
	{
		if(add_def_construct)
			add_constructor< T >();
		if(add_def_copy)
			add_copy_constructor< T >();
	}

	/*-----------------------------------------------------------------
	 * create new instance
	 *----------------------------------------------------------------*/
	// vanilla fucntion pointer as type constructor
	template< typename T, typename... Args >
	void add_constructor(bs_type_ctor_result (*f)(Args&&...)) const {
		creators_[typeid(args_pack< Args... >)] = std::make_pair(
			reinterpret_cast< void(*)() >((creator_callback< Args... >)
				[](void* ff, std::add_lvalue_reference_t< const std::decay_t< Args > >... args) {
					return std::static_pointer_cast< objbase, T >(
						(*static_cast< decltype(f) >(ff))(args...)
					);
				}
			),
			static_cast< void* >(f)
		);
	}

	// std type construction
	// explicit constructor arguments types
	template< typename T, typename... Args >
	void add_constructor() const {
		creators_[typeid(args_pack< Args... >)] = std::make_pair(
			reinterpret_cast< void(*)() >((creator_callback< Args... >)
				[](void*, std::add_lvalue_reference_t< const std::decay_t< Args > >... args) {
					return std::static_pointer_cast< objbase, T >(std::make_shared< T >(args...));
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
			auto callback = reinterpret_cast< creator_callback< Args...> >(creator->second.first);
			return callback(creator->second.second, std::forward< Args >(args)...);
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
	std::string type_name() const {
		return type_name_;
	}
	std::string description() const {
		return description_;
	}
	// TODO: remove this function!
	std::string name() const {
		return type_name_;
	}

	/// tests
	bool is_nil() const {
		return ::blue_sky::is_nil(bs_ti_);
	}
	bool is_copyable() const {
		return (copy_fun_ != nullptr);
	}

	/// conversions
	operator std::string() const {
		return type_name_;
	}
	operator const char*() const {
		return type_name_.c_str();
	}

	//! by default type_descriptors are comparable by bs_type_info
	bool operator <(const type_descriptor& td) const {
		return bs_ti_ < td.bs_ti_;
	}

	//! retrieve type_descriptor of parent class
	type_descriptor parent_td() const {
		if(parent_td_fun_)
			return (*parent_td_fun_)();
		else
			return type_descriptor();
	}
};

// comparison with type string
inline bool operator <(const type_descriptor& td, const std::string& type_string) {
	return (td.type_name() < type_string);
}

inline bool operator ==(const type_descriptor& td, const std::string& type_string) {
	return (td.type_name() == type_string);
}

inline bool operator !=(const type_descriptor& td, const std::string& type_string) {
	return td.type_name() != type_string;
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
							  type_descriptor,
							  type_descriptor,
							  bool >
{
	bool operator()(const type_descriptor& td1, const type_descriptor& td2) const;
};

//  namespace bs {
//
//    template <typename T>
//    std::string 
//    type_name ()
//    {
//      return T::bs_type ().stype_;
//    }
//
//    template <typename T>
//    std::string 
//    type_name (const T &t)
//    {
//      return t.bs_resolve_type ().stype_;
//    }
//
//    template <typename T, bool F>
//    std::string 
//    type_name (const smart_ptr <T, F> &p)
//    {
//      return p->bs_resolve_type ().stype_;
//    }
//
//  } // namespace bs

}	// eof blue_sky namespace

