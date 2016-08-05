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

#include <set>
#include <list>
#include <boost/preprocessor/seq/for_each.hpp>

#define BS_NIL_TYPE_TAG "__blue_sky_nil_type__"

namespace blue_sky {

typedef const std::shared_ptr< objbase >& bs_type_ctor_param;
typedef const std::shared_ptr< objbase >& bs_type_cpy_ctor_param;
typedef objbase* (*BS_TYPE_CREATION_FUN)(bs_type_ctor_param);
typedef objbase* (*BS_TYPE_COPY_FUN)(bs_type_cpy_ctor_param);
typedef blue_sky::type_descriptor (*BS_GET_TD_FUN)();

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
	BS_TYPE_CREATION_FUN creation_fun_;
	BS_TYPE_COPY_FUN copy_fun_;
	BS_GET_TD_FUN parent_td_fun_;
	std::string type_name_; //!< string type name
	std::string description_; //!< Short description of type

	//helper template to retrieve copy function
	template< class T, int nocopy >
	struct extract_copyfn {
		static BS_TYPE_COPY_FUN go() {
			return nullptr;
		}
	};

	template< class T >
	struct extract_copyfn< T, 0 > {
		static BS_TYPE_COPY_FUN go() {
			return T::bs_create_copy;
		}
	};

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

public:
	// default constructor - type_descriptor points to nil
	type_descriptor()
		:
		bs_ti_(nil_type_info()), creation_fun_(nullptr), copy_fun_(nullptr), parent_td_fun_(nullptr),
		type_name_(BS_NIL_TYPE_TAG)
	{}

	// Nil constructor for temporary tasks (searching etc)
	type_descriptor(const std::string& stype)
		:
		bs_ti_(nil_type_info()), creation_fun_(nullptr), copy_fun_(nullptr), parent_td_fun_(nullptr),
		type_name_(stype)
	{}

	// standard constructor
	type_descriptor(
		const BS_TYPE_INFO& ti, const BS_TYPE_CREATION_FUN& cr_fn, const BS_TYPE_COPY_FUN& cp_fn,
		const BS_GET_TD_FUN& parent_td_fn, const char* type_name, const char* description
	) :
		bs_ti_(ti), creation_fun_(cr_fn), copy_fun_(cp_fn), parent_td_fun_(parent_td_fn),
		type_name_(type_name), description_(description)
	{}

	// templated ctor for BlueSky types
	//template< class T, class base = nil, int nocopy = 0 >
	//type_descriptor(Loki::Type2Type< T > /* this_t */, Loki::Type2Type< base > /* base_t */,
	//				Loki::Int2Type< nocopy > /* no_copy_fun */,
	//				const char* type_name = nullptr, const char* description = nullptr)
	//	:
	//	bs_ti_(BS_GET_TI(T)), creation_fun_(T::bs_create_instance),
	//	copy_fun_(extract_copyfn< T, nocopy >::go()),
	//	parent_td_fun_(extract_tdfun< base >::go()),
	//	type_name_(extract_typename< T >::go(type_name)), description_(description)
	//{}

	// templated ctor for BlueSky types
	template< class T, class base = nil, int nocopy = 0, class typename_t = std::nullptr_t >
	type_descriptor(typename_t type_name = nullptr, const char* description = nullptr)
		:
		bs_ti_(BS_GET_TI(T)), creation_fun_(T::bs_create_instance),
		copy_fun_(extract_copyfn< T, nocopy >::go()),
		parent_td_fun_(extract_tdfun< base >::go()),
		type_name_(extract_typename< T >::go(type_name)), description_(description)
	{}

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

