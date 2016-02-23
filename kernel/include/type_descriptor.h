/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Type descriptor class for BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _TYPE_DESCRIPTOR_H
#define _TYPE_DESCRIPTOR_H

#include "bs_common.h"
#include <list>
#include <set>

#include "bs_type_macro.h"
#include "boost/preprocessor/seq/for_each.hpp"

namespace blue_sky {

typedef const smart_ptr< objbase, true >& bs_type_ctor_param;
typedef const smart_ptr< objbase, true >& bs_type_cpy_ctor_param;
typedef objbase* (*BS_TYPE_CREATION_FUN)(bs_type_ctor_param);
typedef objbase* (*BS_TYPE_COPY_FUN)(bs_type_cpy_ctor_param);
typedef blue_sky::type_descriptor (*BS_GET_TD_FUN)();

typedef std::set< smart_ptr< objbase, true > > bs_objinst_holder;
typedef smart_ptr< bs_objinst_holder > sp_objinst;
//typedef const bs_objinst_holder& (*BS_INSTANCES_FUN)();

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
	//BS_INSTANCES_FUN instances_fun_;

	//sp_objinst instances_;
	//void init(const BS_TYPE_OBJ& tp, const BS_OBJECT_CREATION_FUN cr_fn, const BS_OBJECT_COPY_FUN cp_fn,
	//	const std::string& stype, const std::string& short_descr, const std::string& long_descr);

	//helper template to retrieve copy function
	template< class T, int nocopy >
	struct extract_copyfn {
		static BS_TYPE_COPY_FUN go() {
			return NULL;
		}
	};

	template< class T >
	struct extract_copyfn< T, 0 > {
		static BS_TYPE_COPY_FUN go() {
			return T::bs_create_copy;
		}
	};

public:
	std::string stype_; //!< Type string
	std::string short_descr_; //!< Short description of type
	std::string long_descr_; //!< Long description of type

	//default constructor - type_descriptor points to nil
	type_descriptor();

	//standard constructor
	type_descriptor(const BS_TYPE_INFO& ti, const BS_TYPE_CREATION_FUN& cr_fn, const BS_TYPE_COPY_FUN& cp_fn,
		const BS_GET_TD_FUN& parent_td_fn, const std::string& stype, const std::string& short_descr,
		const std::string& long_descr = "");

	//templated ctor
	template< class T, class base, int nocopy >
	type_descriptor(Loki::Type2Type< T > /* this_t */, Loki::Type2Type< base > /* base_t */,
					Loki::Int2Type< nocopy > /* no_copy_fun */,
					const std::string& stype, const std::string& short_descr, const std::string& long_descr = "")
		: bs_ti_(BS_GET_TI(T)), creation_fun_(T::bs_create_instance),
		  copy_fun_(extract_copyfn< T, nocopy >::go()), parent_td_fun_(base::bs_type),
		  stype_(stype), short_descr_(short_descr), long_descr_(long_descr)
	{}

	//Nil descriptor constructor for temporary tasks (searching etc)
	type_descriptor(const std::string& stype);

	//DEBUG
	//~type_descriptor();
	//operator =
	//type_descriptor& operator =(const type_descriptor& td);

	//type_info accessor
	BS_TYPE_INFO type() const {
		return bs_ti_;
	};

	bool is_nil() const {
		return bs_ti_.is_nil();
	}

	bool is_copyable() const {
		return (copy_fun_ != NULL);
	}

	//! \brief conversion to string returns type string
	operator std::string() const {
		return stype_;
	}
	//! \brief conversion to const char*
	operator const char*() const {
		return stype_.c_str();
	}

  std::string 
  name () const
  {
    return stype_;
  }

	//! by default type_descriptors are comparable by bs_type_info
	bool operator <(const type_descriptor& td) const {
		return bs_ti_ < td.bs_ti_;
	}

	//! retrieve type_descriptor of parent class
	type_descriptor parent_td() const;

	//! instances access
//		bs_objinst_holder::const_iterator inst_begin() const {
//			return instances_->begin();
//		}
//
//		bs_objinst_holder::const_iterator inst_end() const {
//			return instances_->end();
//		}
//
//		ulong inst_cnt() const {
//			return static_cast< ulong >(instances_->size());
//		}
};

//comparison with type string
BS_API inline bool operator <(const type_descriptor& td, const std::string& type_string) {
	return (td.stype_ < type_string);
}

BS_API inline bool operator ==(const type_descriptor& td, const std::string& type_string) {
	return (td.stype_ == type_string);
}

BS_API inline bool operator !=(const type_descriptor& td, const std::string& type_string) {
	return !(td == type_string);
}

//comparison with bs_type_info
BS_API inline bool operator <(const type_descriptor& td, const BS_TYPE_INFO& ti) {
	return (td.type() < ti);
}

BS_API inline bool operator ==(const type_descriptor& td, const BS_TYPE_INFO& ti) {
	return (td.type() == ti);
}

BS_API inline bool operator !=(const type_descriptor& td, const BS_TYPE_INFO& ti) {
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

  namespace bs {

    template <typename T>
    std::string 
    type_name ()
    {
      return T::bs_type ().stype_;
    }

    template <typename T>
    std::string 
    type_name (const T &t)
    {
      return t.bs_resolve_type ().stype_;
    }

    template <typename T, bool F>
    std::string 
    type_name (const smart_ptr <T, F> &p)
    {
      return p->bs_resolve_type ().stype_;
    }

  } // namespace bs

}	//blue_sky namespace

#endif
