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

#include "boost/preprocessor/cat.hpp"
#include "boost/preprocessor/punctuation/comma_if.hpp"
#include "boost/preprocessor/control/iif.hpp"
#include "boost/preprocessor/facilities/empty.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"
#include "boost/preprocessor/seq/enum.hpp"
#include "boost/preprocessor/seq/for_each_i.hpp"
#include "boost/preprocessor/seq/size.hpp"

//================================ macro definitions ===================================================================
#define BS_TYPE_DECL \
public: static blue_sky::type_descriptor bs_type(); \
virtual blue_sky::type_descriptor bs_resolve_type() const; \
private: friend class blue_sky::type_descriptor; \
static objbase* bs_create_instance(bs_type_ctor_param param = NULL); \
static objbase* bs_create_copy(bs_type_cpy_ctor_param param = NULL);

#define BS_TYPE_DECL_MEM(T, base, type_string, short_descr, long_descr, nocopy)			\
public:																																							\
	static blue_sky::type_descriptor bs_type()																				\
	{																																									\
		return BS_TD_IMPL(T, base, type_string, short_descr, long_descr, nocopy);				\
	}																																									\
	virtual blue_sky::type_descriptor bs_resolve_type() const													\
	{																																									\
		return bs_type ();																															\
	}																																									\
private: 																																						\
	friend class blue_sky::type_descriptor;

#define BS_TYPE_DECL_T_MEM_(T, base, stype_prefix, short_descr, long_descr, nocopy) \
public: static blue_sky::type_descriptor bs_type(); \
virtual blue_sky::type_descriptor bs_resolve_type() const { return bs_type(); } \
private: friend class type_descriptor; \
static const type_descriptor& td_maker(const std::string& stype_postfix) { \
	static blue_sky::type_descriptor td(Loki::Type2Type< T >(), Loki::Type2Type< base >(), Loki::Int2Type< nocopy >(), \
		std::string(stype_prefix) + stype_postfix, short_descr, long_descr); \
	return td; \
}

#define BS_TYPE_DECL_T_MEM(T, base, stype_prefix, short_descr, long_descr) \
BS_TYPE_DECL_T_MEM_(T, base, stype_prefix, short_descr, long_descr, false)

#define BS_TYPE_DECL_T_MEM_NOCOPY(T, base, stype_prefix, short_descr, long_descr) \
BS_TYPE_DECL_T_MEM_(T, base, stype_prefix, short_descr, long_descr, true)

//----------------- common implementation ------------------------------------------------------------------------------
#define BS_TD_IMPL(T, base, type_string, short_descr, long_descr, nocopy) \
blue_sky::type_descriptor(Loki::Type2Type< BOOST_PP_SEQ_ENUM(T) >(), Loki::Type2Type< BOOST_PP_SEQ_ENUM(base) >(), \
Loki::Int2Type< nocopy >(), type_string, short_descr, long_descr)

#define BS_TYPE_IMPL_EXT_(prefix, T, base, type_string, short_descr, long_descr, nocopy) \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::type_descriptor BOOST_PP_SEQ_ENUM(T)::bs_type() \
	{ return BS_TD_IMPL(T, base, type_string, short_descr, long_descr, nocopy); } \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::type_descriptor BOOST_PP_SEQ_ENUM(T)::bs_resolve_type() const \
	{ return bs_type(); }

//----------------- implementation for non-templated classes -----------------------------------------------------------
#define BS_TYPE_IMPL(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((), (T), (base), type_string, short_descr, long_descr, false)
/*blue_sky::type_descriptor T::bs_type() { return BS_TD_IMPL(T, base, type_string, short_descr, long_descr, false); } \
blue_sky::type_descriptor T::bs_resolve_type() const { return bs_type(); } */

#define BS_TYPE_IMPL_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((), (T), (base), type_string, short_descr, long_descr, true)
/*blue_sky::type_descriptor T::bs_type() { return BS_TD_IMPL(T, base, type_string, short_descr, long_descr, true); } \
blue_sky::type_descriptor T::bs_resolve_type() const { return bs_type(); }*/

#define BS_TYPE_IMPL_SHORT(T, short_descr) \
BS_TYPE_IMPL(T, #T, short_descr, "")

#define BS_TYPE_IMPL_NOCOPY_SHORT(T, short_descr) \
BS_TYPE_IMPL_NOCOPY(T, #T, short_descr, "")

//------------------- templated implementation I - creates specializations ---------------------------------------------
#define BS_TYPE_IMPL_T(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), (T), (base), type_string, short_descr, long_descr, false)
/*template< > BS_API_PLUGIN blue_sky::type_descriptor T::bs_type() { \
	return BS_TD_IMPL(T, base, type_string, short_descr, long_descr, false); } \
template< > BS_API_PLUGIN blue_sky::type_descriptor T::bs_resolve_type() const { return bs_type(); } */

#define BS_TYPE_IMPL_T_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), (T), (base), type_string, short_descr, long_descr, true)
/*template< > BS_API_PLUGIN blue_sky::type_descriptor T::bs_type() { \
	return BS_TD_IMPL_NOCOPY(T, base, type_string, short_descr, long_descr, true); } \
template< > BS_API_PLUGIN blue_sky::type_descriptor T::bs_resolve_type() const { return bs_type(); }*/

//! put your class specification as well as base's specification in round braces!
#define BS_TYPE_IMPL_T_EXT(T_tup_size, T_tup, base_tup_size, base_tup, type_string, short_descr, long_descr, nocopy) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T_tup_size, T_tup), \
BOOST_PP_TUPLE_TO_SEQ(base_tup_size, base_tup), type_string, short_descr, long_descr, nocopy)

#define BS_TYPE_IMPL_T_SHORT(T, short_descr) \
BS_TYPE_IMPL_T(T, #T, short_descr, "")

#define BS_TYPE_IMPL_T_NOCOPY_SHORT(T, short_descr) \
BS_TYPE_IMPL_T_NOCOPY(T, #T, short_descr, "")

//------------------- templated implementation II - creates definition of bs_type --------------------------------------
#define BS_TYPE_IMPL_T_MEM(T, spec_type) \
template< > blue_sky::type_descriptor T< spec_type >::bs_type() { \
	return td_maker(std::string("_") + #spec_type); }

//------------------- common extended create & copy instance macroses --------------------------------------------------
#define BS_TYPE_STD_CREATE_EXT_(prefix, T, is_decl) \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::objbase* \
BOOST_PP_SEQ_ENUM(BOOST_PP_IIF(is_decl, (), T))BOOST_PP_IIF(is_decl, BOOST_PP_EMPTY(), ::)\
bs_create_instance(bs_type_ctor_param param BOOST_PP_IIF(is_decl, = NULL, BOOST_PP_EMPTY())) \
{ return new BOOST_PP_SEQ_ENUM(T)(param); }

#define BS_TYPE_STD_COPY_EXT_(prefix, T, is_decl) \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::objbase* BOOST_PP_SEQ_ENUM(BOOST_PP_IIF(is_decl, (), T))BOOST_PP_IIF(is_decl, BOOST_PP_EMPTY(), ::)\
bs_create_copy(bs_type_cpy_ctor_param src) { \
	return new BOOST_PP_SEQ_ENUM(T)(*static_cast< const BOOST_PP_SEQ_ENUM(T)* >(src.get())); \
}

//------------------- bs_create_instance macro -------------------------------------------------------------------------
#define BLUE_SKY_TYPE_STD_CREATE(T) \
BS_TYPE_STD_CREATE_EXT_((), (T), 0)
/*blue_sky::objbase* T::bs_create_instance(bs_type_ctor_param param) { \
	return new T(param); \
}*/

#define BLUE_SKY_TYPE_STD_CREATE_MEM(T) \
	BS_TYPE_STD_CREATE_EXT_((static), (T), 1)

#define BLUE_SKY_TYPE_STD_CREATE_T(T) \
BS_TYPE_STD_CREATE_EXT_((template< > BS_API_PLUGIN), (T), 0)
/*template< > BS_API_PLUGIN \
blue_sky::objbase* T::bs_create_instance(bs_type_ctor_param param) { \
	return new T(param); \
}*/

//! put full specialization T in round braces
#define BLUE_SKY_TYPE_STD_CREATE_T_EXT(T) \
BS_TYPE_STD_CREATE_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T), 0)

//generates bs_create_instance as member function
#define BLUE_SKY_TYPE_STD_CREATE_T_MEM(T) \
BS_TYPE_STD_CREATE_EXT_((public: static), (T), 1)
/*
public: static blue_sky::objbase* bs_create_instance(bs_type_ctor_param param = NULL) { \
	return new T(param); \
}*/

//--------- extended create instance generator for templated classes with any template parameters number ---------------
//some helper macro
#define BS_CLASS_CHOKER(r, data, i, elem) BOOST_PP_COMMA_IF(i) BOOST_PP_CAT(A, i)
#define BS_CLIST_FORMER(tp_seq) BOOST_PP_SEQ_FOR_EACH_I(BS_CLASS_CHOKER, _, tp_seq)

#define BS_TLIST_NAMER(r, data, i, elem) BOOST_PP_COMMA_IF(i) elem BOOST_PP_CAT(A, i)
#define BS_TLIST_FORMER(tp_seq) BOOST_PP_SEQ_FOR_EACH_I(BS_TLIST_NAMER, _, tp_seq)

//! surround template params list with round braces
#define BLUE_SKY_TYPE_STD_CREATE_T_DEF(T, t_params) BS_TYPE_STD_CREATE_EXT_( \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

//----------------- bs_create_copy macro -------------------------------------------------------------------------------
#define BLUE_SKY_TYPE_STD_COPY(T) \
BS_TYPE_STD_COPY_EXT_((), (T), 0)

#define BLUE_SKY_TYPE_STD_COPY_MEM(T) \
	BS_TYPE_STD_COPY_EXT_((static), (T), 1)
/*
blue_sky::objbase* T::bs_create_copy(bs_type_cpy_ctor_param src) { \
	return new T(*static_cast< const T* >(src.get())); \
}*/

#define BLUE_SKY_TYPE_STD_COPY_T(T) \
BS_TYPE_STD_COPY_EXT_((template< > BS_API_PLUGIN), (T), 0)
/*template< > BS_API_PLUGIN \
blue_sky::objbase* T::bs_create_copy(bs_type_cpy_ctor_param src) { \
	return new T(*static_cast< const T* >(src.get())); \
}*/

#define BLUE_SKY_TYPE_STD_COPY_T_MEM(T) \
BS_TYPE_STD_COPY_EXT_((public: static), (T), 1)
/*
public: static blue_sky::objbase* bs_create_copy(bs_type_cpy_ctor_param src) { \
	return new T(*static_cast< const T* >(src.get())); \
}*/

//! surround template params list with round braces
#define BLUE_SKY_TYPE_STD_COPY_T_DEF(T, t_params) BS_TYPE_STD_COPY_EXT_( \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

//================================ end of macro ========================================================================

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
}

#endif
