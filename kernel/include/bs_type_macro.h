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

/// @file bs_type_macro.h
/// @brief Macro definitions for automatic type_descriptor maintance in BlueSky objects
/// @author Alexander Gagarin aka uentity
/// @date 2009-06-26

#ifndef _BS_TYPE_MACRO_H
#define _BS_TYPE_MACRO_H

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

#define BS_TYPE_DECL_MEM(T, base, type_string, short_descr, long_descr, nocopy)   \
public:                                                                           \
    static blue_sky::type_descriptor bs_type()                                    \
    {                                                                             \
        return BS_TD_IMPL(T, base, type_string, short_descr, long_descr, nocopy); \
    }                                                                             \
    virtual blue_sky::type_descriptor bs_resolve_type() const                     \
    {                                                                             \
        return bs_type ();                                                        \
    }                                                                             \
private:                                                                          \
    friend class blue_sky::type_descriptor;

#define BS_TYPE_DECL_T_MEM_(T, base, stype_prefix, short_descr, long_descr, nocopy)                                    \
public: static blue_sky::type_descriptor bs_type();                                                                    \
virtual blue_sky::type_descriptor bs_resolve_type() const { return bs_type(); }                                        \
private: friend class type_descriptor;                                                                                 \
static const type_descriptor& td_maker(const std::string& stype_postfix) {                                             \
    static blue_sky::type_descriptor td(Loki::Type2Type< T >(), Loki::Type2Type< base >(), Loki::Int2Type< nocopy >(), \
        std::string(stype_prefix) + stype_postfix, short_descr, long_descr);                                           \
    return td;                                                                                                         \
}

#define BS_TYPE_DECL_T_MEM(T, base, stype_prefix, short_descr, long_descr) \
BS_TYPE_DECL_T_MEM_(T, base, stype_prefix, short_descr, long_descr, false)

#define BS_TYPE_DECL_T_MEM_NOCOPY(T, base, stype_prefix, short_descr, long_descr) \
BS_TYPE_DECL_T_MEM_(T, base, stype_prefix, short_descr, long_descr, true)

//----------------- common implementation ------------------------------------------------------------------------------
#define BS_TD_IMPL(T, base, type_string, short_descr, long_descr, nocopy)                                          \
blue_sky::type_descriptor(Loki::Type2Type< BOOST_PP_SEQ_ENUM(T) >(), Loki::Type2Type< BOOST_PP_SEQ_ENUM(base) >(), \
Loki::Int2Type< nocopy >(), type_string, short_descr, long_descr)

#define BS_TYPE_IMPL_EXT_(prefix, T, base, type_string, short_descr, long_descr, nocopy)          \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::type_descriptor BOOST_PP_SEQ_ENUM(T)::bs_type()               \
    { return BS_TD_IMPL(T, base, type_string, short_descr, long_descr, nocopy); }                 \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::type_descriptor BOOST_PP_SEQ_ENUM(T)::bs_resolve_type() const \
    { return bs_type(); }

//----------------- implementation for non-templated classes -----------------------------------------------------------
#define BS_TYPE_IMPL(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((), (T), (base), type_string, short_descr, long_descr, false)

#define BS_TYPE_IMPL_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((), (T), (base), type_string, short_descr, long_descr, true)

#define BS_TYPE_IMPL_SHORT(T, short_descr) \
BS_TYPE_IMPL(T, #T, short_descr, "")

#define BS_TYPE_IMPL_NOCOPY_SHORT(T, short_descr) \
BS_TYPE_IMPL_NOCOPY(T, #T, short_descr, "")

//------------------- templated implementation I - creates specializations ---------------------------------------------
#define BS_TYPE_IMPL_T(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), (T), (base), type_string, short_descr, long_descr, false)

#define BS_TYPE_IMPL_T_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), (T), (base), type_string, short_descr, long_descr, true)

//! put your class specification as well as base's specification in round braces!
#define BS_TYPE_IMPL_T_EXT(T_tup_size, T_tup, base_tup_size, base_tup, type_string, short_descr, long_descr, nocopy) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T_tup_size, T_tup),                             \
BOOST_PP_TUPLE_TO_SEQ(base_tup_size, base_tup), type_string, short_descr, long_descr, nocopy)

#define BS_TYPE_IMPL_T_SHORT(T, short_descr) \
BS_TYPE_IMPL_T(T, #T, short_descr, "")

#define BS_TYPE_IMPL_T_NOCOPY_SHORT(T, short_descr) \
BS_TYPE_IMPL_T_NOCOPY(T, #T, short_descr, "")

//------------------- templated implementation II - creates definition of bs_type --------------------------------------
#define BS_TYPE_IMPL_T_MEM(T, spec_type)                          \
template< > blue_sky::type_descriptor T< spec_type >::bs_type() { \
    return td_maker(std::string("_") + #spec_type); }

//------------------- common extended create & copy instance macroses --------------------------------------------------
#define BS_TYPE_STD_CREATE_EXT_(prefix, T, is_decl)                                          \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::objbase*                                                 \
BOOST_PP_SEQ_ENUM(BOOST_PP_IIF(is_decl, (), T))BOOST_PP_IIF(is_decl, BOOST_PP_EMPTY(), ::)   \
bs_create_instance(bs_type_ctor_param param BOOST_PP_IIF(is_decl, = NULL, BOOST_PP_EMPTY())) \
{ return new BOOST_PP_SEQ_ENUM(T)(param); }

#define BS_TYPE_STD_COPY_EXT_(prefix, T, is_decl)                                                                                       \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::objbase* BOOST_PP_SEQ_ENUM(BOOST_PP_IIF(is_decl, (), T))BOOST_PP_IIF(is_decl, BOOST_PP_EMPTY(), ::) \
bs_create_copy(bs_type_cpy_ctor_param src) {                                                                                            \
    return new BOOST_PP_SEQ_ENUM(T)(*static_cast< const BOOST_PP_SEQ_ENUM(T)* >(src.get()));                                            \
}

//------------------- bs_create_instance macro -------------------------------------------------------------------------
#define BLUE_SKY_TYPE_STD_CREATE(T) \
BS_TYPE_STD_CREATE_EXT_((), (T), 0)

#define BLUE_SKY_TYPE_STD_CREATE_MEM(T) \
	BS_TYPE_STD_CREATE_EXT_((static), (T), 1)

#define BLUE_SKY_TYPE_STD_CREATE_T(T) \
BS_TYPE_STD_CREATE_EXT_((template< > BS_API_PLUGIN), (T), 0)

//! put full specialization T in round braces
#define BLUE_SKY_TYPE_STD_CREATE_T_EXT(T) \
BS_TYPE_STD_CREATE_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T), 0)

//generates bs_create_instance as member function
#define BLUE_SKY_TYPE_STD_CREATE_T_MEM(T) \
BS_TYPE_STD_CREATE_EXT_((public: static), (T), 1)

//--------- extended create instance generator for templated classes with any template parameters number ---------------
//some helper macro
#define BS_CLIST_CHOKER(r, data, i, elem) BOOST_PP_COMMA_IF(i) BOOST_PP_CAT(A, i)
#define BS_CLIST_FORMER(tp_seq) BOOST_PP_SEQ_FOR_EACH_I(BS_CLIST_CHOKER, _, tp_seq)

#define BS_TLIST_NAMER(r, data, i, elem) BOOST_PP_COMMA_IF(i) elem BOOST_PP_CAT(A, i)
#define BS_TLIST_FORMER(tp_seq) BOOST_PP_SEQ_FOR_EACH_I(BS_TLIST_NAMER, _, tp_seq)

//! surround template params list with round braces
#define BLUE_SKY_TYPE_STD_CREATE_T_DEF(T, t_params) BS_TYPE_STD_CREATE_EXT_(                 \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

//----------------- bs_create_copy macro -------------------------------------------------------------------------------
#define BLUE_SKY_TYPE_STD_COPY(T) \
BS_TYPE_STD_COPY_EXT_((), (T), 0)

#define BLUE_SKY_TYPE_STD_COPY_MEM(T) \
BS_TYPE_STD_COPY_EXT_((static), (T), 1)

#define BLUE_SKY_TYPE_STD_COPY_T(T) \
BS_TYPE_STD_COPY_EXT_((template< > BS_API_PLUGIN), (T), 0)

#define BLUE_SKY_TYPE_STD_COPY_T_MEM(T) \
BS_TYPE_STD_COPY_EXT_((public: static), (T), 1)

//! surround template params list with round braces
#define BLUE_SKY_TYPE_STD_COPY_T_DEF(T, t_params) BS_TYPE_STD_COPY_EXT_(                     \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

#endif	// guard

