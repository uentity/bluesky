/// @file
/// @author uentity
/// @date 07.10.2016
/// @brief Macro definitions for automatic type_descriptor maintance in BlueSky objects
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/logical/compl.hpp>

#include <boost/preprocessor/seq/for_each_i.hpp>

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/tuple/enum.hpp>

// trick to overcome M$VC c4003 warnings
//#include <boost/preprocessor/array/data.hpp>
//#include <boost/preprocessor/tuple/rem.hpp>

/*-----------------------------------------------------------------
 * helpers
 *----------------------------------------------------------------*/
#define BS_SEQ_NIL() (;)

#define BS_FMT_TYPE_SPEC(T, is_decl) \
BOOST_PP_TUPLE_ENUM(BOOST_PP_IIF(is_decl, (), T))BOOST_PP_EXPR_IIF(BOOST_PP_COMPL(is_decl), ::)

// helper to generate templated type specification
// from type name T and tuple of template arguments
//#define BS_CLIST_CHOKER(r, data, i, elem) BOOST_PP_COMMA_IF(i) BOOST_PP_CAT(A, i)
//#define BS_CLIST_FORMER(tp_seq) BOOST_PP_SEQ_FOR_EACH_I(BS_CLIST_CHOKER, _, tp_seq)
//
//#define BS_TLIST_NAMER(r, data, i, elem) BOOST_PP_COMMA_IF(i) elem BOOST_PP_CAT(A, i)
//#define BS_TLIST_FORMER(tp_seq) BOOST_PP_SEQ_FOR_EACH_I(BS_TLIST_NAMER, _, tp_seq)


/*-----------------------------------------------------------------
 * declarations
 *----------------------------------------------------------------*/
// only declare functions, bs_resolve_type() is always inline
#define BS_TYPE_DECL                                       \
BS_RESOLVE_TYPE_IMPL_INL                                   \
public: static const blue_sky::type_descriptor& bs_type(); \
private: friend class blue_sky::type_descriptor;

// inline standard implementation that automates type name generation for templated types
// type name will be constructed in form: "prefix spectype1 spectype2 ...", where
// prefix can be specified by user and name tail is concatenation of spec types
// this macro is intended to be used with multiple BS_TYPE_IMPL_INL_EXT for each specialization
// in cpp file
#define BS_TYPE_DECL_INL_BEGIN(T, base, type_name_prefix, descr, add_def_create, add_def_copy)           \
BS_RESOLVE_TYPE_IMPL_INL                                                                                 \
public: static const blue_sky::type_descriptor& bs_type();                                               \
private: friend class blue_sky::type_descriptor;                                                         \
static const blue_sky::type_descriptor& td_maker(const std::string& tname_postfix) {                     \
    static blue_sky::type_descriptor td(                                                                 \
        identity< T >(), identity< base >(),                                                             \
        std::string(type_name_prefix) + tname_postfix, descr,                                            \
        std::integral_constant< bool, add_def_create >(), std::integral_constant< bool, add_def_copy >() \
    );
// you can additionally modify td between BEGIN and END, for ex. add constructor
// td.add_constructor< ... >(...)
#define BS_TYPE_DECL_INL_END return td; }

#define BS_TYPE_DECL_INL(T, base, type_name_prefix, descr, add_def_create, add_def_copy) \
BS_TYPE_DECL_INL_BEGIN(T, base, type_name_prefix, descr, add_def_create, add_def_copy)   \
BS_TYPE_DECL_INL_END

/*-----------------------------------------------------------------
 * bs_type() implementation
 *----------------------------------------------------------------*/
// assume that types T and base are passed as tuples
#define BS_TD_IMPL(T_tup, base_tup, type_name, descr, add_def_create, add_def_copy)                    \
blue_sky::type_descriptor td(blue_sky::identity< BOOST_PP_TUPLE_ENUM(T_tup) >(),                       \
    blue_sky::identity< BOOST_PP_TUPLE_ENUM(base_tup) >(), type_name, descr,                           \
    std::integral_constant< bool, add_def_create >(), std::integral_constant< bool, add_def_copy >());
// prefix is also a tuple
#define BS_TYPE_IMPL_(prefix_tup, T_tup, base_tup, type_name, descr, add_def_create, add_def_copy, is_decl) \
BOOST_PP_TUPLE_ENUM(prefix_tup) const blue_sky::type_descriptor& BS_FMT_TYPE_SPEC(T_tup, is_decl) bs_type() \
{ static BS_TD_IMPL(T_tup, base_tup, type_name, descr, add_def_create, add_def_copy); return td; }

// here T ans base are type names, not tuples
// implementation for non-template types
#define BS_TYPE_IMPL(T, base, type_name, descr, add_def_create, add_def_copy) \
BS_TYPE_IMPL_(BS_SEQ_NIL(), (T), (base), type_name, descr, add_def_create, add_def_copy, 0)
// for T and base with 1 template parameter
// T and base are template specializations, not tuples
#define BS_TYPE_IMPL_T1(T, base, type_name, descr, add_def_create, add_def_copy) \
BS_TYPE_IMPL_((template< > BS_API_PLUGIN), (T), (base), type_name, descr, add_def_create, add_def_copy)
// most generic macro taking specializations of T and base as tuples
// for templates with > 1 template parameters
#define BS_TYPE_IMPL_T(T, T_spec_tup, base, base_spec_tup, type_name, descr, add_def_create, add_def_copy) \
BS_TYPE_IMPL_((template< > BS_API_PLUGIN), (T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >),                         \
    (base< BOOST_PP_TUPLE_ENUM(base_spec_tup) >), type_name, descr, add_def_create, add_def_copy, 0)

/*-----------------------------------------------------------------
 * bs_type() implementation for templated types with auto-generated typename
 *----------------------------------------------------------------*/
#define _OP(r, data, x) x

#if defined(_MSC_VER)

// NOTE: it seems that VS compiler doesn't instantiate static template class members
// even when class is explicitly instantiated.
// So we need to add explicit instantiation for static member functions, declared in
// class body.
#define BS_TYPE_IMPL_INL_T(T, T_spec_tup)                                                    \
template< > BS_API_PLUGIN const blue_sky::type_descriptor&                                   \
T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >::bs_type()                                              \
{ return td_maker(std::string(" ") +                                                         \
    BOOST_PP_STRINGIZE(BOOST_PP_SEQ_FOR_EACH(_OP, _, BOOST_PP_TUPLE_TO_SEQ(T_spec_tup)))); } \
template BS_API_PLUGIN const blue_sky::type_descriptor&                                      \
T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >::td_maker(const std::string&);

#else

#define BS_TYPE_IMPL_INL_T(T, T_spec_tup)                                                    \
template< > BS_API_PLUGIN const blue_sky::type_descriptor&                                   \
T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >::bs_type()                                              \
{ return td_maker(std::string(" ") +                                                         \
    BOOST_PP_STRINGIZE(BOOST_PP_SEQ_FOR_EACH(_OP, _, BOOST_PP_TUPLE_TO_SEQ(T_spec_tup)))); }

#endif

// for templates with 1 parameter, T is not a tuple
// NOTE! This macro is an exception where you should pass single spec_type as second parameter
#define BS_TYPE_IMPL_INL_T1(T, spec_type) \
BS_TYPE_IMPL_INL_T(T, (spec_type))

/*-----------------------------------------------------------------
 * bs_resolve_type() implementation - calls bs_type()
 *----------------------------------------------------------------*/
// prefix and T are tuples
#define BS_RESOLVE_TYPE_IMPL_(prefix_tup, T_tup, is_decl)                                         \
BOOST_PP_TUPLE_ENUM(prefix_tup) const blue_sky::type_descriptor& BS_FMT_TYPE_SPEC(T_tup, is_decl) \
bs_resolve_type() const { return bs_type(); }

// here T ans base are type names, not tuples
// inline bs_resolve_type() implementation
#define BS_RESOLVE_TYPE_IMPL_INL \
BS_RESOLVE_TYPE_IMPL_((public:), BS_SEQ_NIL(), 1)

// bs_resolce_type() is inlined in *_DECL macro family
// so the following macro are included only to make system more "complete"
// implementation for non-template types
#define BS_RESOLVE_TYPE_IMPL(T) \
BS_RESOLVE_TYPE_IMPL_(BS_SEQ_NIL(), (T), 0)
// for type T with 1 template parameter
// T is template specializations, not tuple
#define BS_RESOLVE_TYPE_IMPL_T1(T) \
BS_RESOLVE_TYPE_IMPL_((template< >), (T), 0)
// most generic macro taking T specialization as tuple
// for templates with > 1 template parameters
#define BS_RESOLVE_TYPE_IMPL_T(T, T_spec_tup) \
BS_RESOLVE_TYPE_IMPL_((template< >), (T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >), 0)

/*-----------------------------------------------------------------
 * auto-register type constructors
 *----------------------------------------------------------------*/
// create unqiue static int variables
// registration code is executed during variables initialization
// pass constructor arguments types as tuple
// ctor_has_args flag is needed, because empty tuple is considered to have size = 1
// so we need to additionally indicate that constructor have > 0 agruments
#define BS_TYPE_ADD_CONSTRUCTOR_(T_tup, ctor_args_tup, ctor_has_args, f_tup)           \
namespace {                                                                            \
static int BOOST_PP_CAT(_bs_reg_create_, __LINE__) = [](){                             \
    BOOST_PP_TUPLE_ENUM(T_tup)::bs_type().add_constructor<                             \
    BOOST_PP_TUPLE_ENUM(T_tup) BOOST_PP_COMMA_IF(ctor_has_args)                        \
    BOOST_PP_TUPLE_ENUM(ctor_args_tup) >(BOOST_PP_TUPLE_ENUM(f_tup)); return 0; }(); }

// for simple types (<= 1 template params) ctors with non-zero arguments
#define BS_TYPE_ADD_CONSTRUCTOR(T, ctor_args_tup) \
BS_TYPE_ADD_CONSTRUCTOR_((T), ctor_args_tup, 1, ())
// add default ctor with no arguments
#define BS_TYPE_ADD_DEF_CONSTRUCTOR(T) \
BS_TYPE_ADD_CONSTRUCTOR_((T), (), 0, ())

// for templated types (> 1 template params) ctors with non-zero arguments
#define BS_TYPE_ADD_CONSTRUCTOR_T(T, T_spec_tup, ctor_args_tup) \
BS_TYPE_ADD_CONSTRUCTOR_((T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >), ctor_args_tup, 1, ())
// add default ctor with no arguments
#define BS_TYPE_ADD_DEF_CONSTRUCTOR_T(T, T_spec_tup) \
BS_TYPE_ADD_CONSTRUCTOR_((T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >), (), 0)

// add constructor as free function
#define BS_TYPE_ADD_CONSTRUCTOR_F_(T_tup, f_tup)                                         \
namespace {                                                                              \
static int BOOST_PP_CAT(_bs_reg_create_, __LINE__) = [](){                               \
    BOOST_PP_TUPLE_ENUM(T_tup)::bs_type().add_constructor< BOOST_PP_TUPLE_ENUM(T_tup) >( \
    BOOST_PP_TUPLE_ENUM(f_tup)); return 0; }(); }

// for simple types and factory functions (<= 1 template params)
#define BS_TYPE_ADD_CONSTRUCTOR_F(T, f) \
BS_TYPE_ADD_CONSTRUCTOR_F_((T), (f))
// for templated types and factory functions (> 1 template params)
// NOTE: assume that factory function is also templated!
#define BS_TYPE_ADD_CONSTRUCTOR_T_F(T, T_spec_tup, f, f_spec_tup) \
BS_TYPE_ADD_CONSTRUCTOR_F_((T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >), (f< BOOST_PP_TUPLE_ENUM(f_spec_tup) >))

/*-----------------------------------------------------------------
 * auto-register type copy constructors
 *----------------------------------------------------------------*/
#define BS_TYPE_ADD_COPY_CONSTRUCTOR_(T_tup)                     \
namespace {                                                      \
static int BOOST_PP_CAT(_bs_reg_copy_, BOOST_PP_COUNTER) = [](){ \
    BOOST_PP_TUPLE_ENUM(T_tup)::bs_type().add_copy_constructor<  \
    BOOST_PP_TUPLE_ENUM(T_tup) >(); return 0; }(); }

// for simple types (<= 1 template params)
// T is not a tuple
#define BS_TYPE_ADD_COPY_CONSTRUCTOR(T) \
BS_TYPE_ADD_COPY_CONSTRUCTOR_((T))

// for templated types (> 1 template params) ctors with non-zero arguments
#define BS_TYPE_ADD_COPY_CONSTRUCTOR_T(T, T_spec_tup) \
BS_TYPE_ADD_COPY_CONSTRUCTOR_((T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >))

/*-----------------------------------------------------------------
 * auto-register type in BS kernel
 *----------------------------------------------------------------*/
// type passed as tuple
#define BS_REGISTER_TYPE_(T_tup)                                            \
extern "C" const ::blue_sky::plugin_descriptor* bs_get_plugin_descriptor(); \
namespace { static int BOOST_PP_CAT(_bs_reg_type_, __LINE__) =              \
[]() { ::blue_sky::give_kernel::Instance().register_type(                   \
    BS_FMT_TYPE_SPEC(T_tup, 0) bs_type(), bs_get_plugin_descriptor());      \
    return 0; }(); }

// non-templated types and types with 1 template parameter
#define BS_REGISTER_TYPE(T) \
BS_REGISTER_TYPE_((T))

// templated type with > 1 parameters
// pass specialization type as tuple
#define BS_REGISTER_TYPE_T(T, T_spec_tup) \
BS_REGISTER_TYPE_((T< BOOST_PP_TUPLE_ENUM(T_spec_tup) >))

