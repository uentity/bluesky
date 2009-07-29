// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#ifndef _TYPE_DESCRIPTOR_H
#define _TYPE_DESCRIPTOR_H

#include "bs_common.h"
#include <list>
#include <set>

#include "bs_type_macro.h"

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

}	//blue_sky namespace

#endif
