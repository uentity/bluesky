/// @file
/// @author uentity
/// @date 15.03.2017
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/logical/not.hpp>

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/enum.hpp>

#include <boost/preprocessor/seq/for_each_i.hpp>

// [NOTE] BS serialization macroses start with `BSS_` prefix

/*-----------------------------------------------------------------
 * helpers to generate type
 *----------------------------------------------------------------*/
#define BS_TPL_ARG_(r, param, i, prefix_elem) \
BOOST_PP_COMMA_IF(i) prefix_elem BOOST_PP_CAT(param, i)

#define BS_TYPE_TPL_ARG_(z, i, param) \
BOOST_PP_CAT(param, i)

#define BS_ECHO_(z, i, text) text

/////////////////////////////////////////////////////////////////
//  helper to test for empty tuple/sequence/variadic args/...
//  solves issue when BOOST_PP_TUPLE_SIZE returns 1 even for empty `()` tuple
//  (same applies to sequence, enum, var args, etc... and caused by nature of C99 preprocessor

// 1. Define helper that detects if comma exists in arguments list
#define _ARG24(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, ...) _23
#define HAS_COMMA(...) _ARG24(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
// _ISEMPTY(...) expands to HAS_COMMA(_IS_EMPTY_CASE_xxxx)
#define PASTE5(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
#define _ISEMPTY(_0, _1, _2, _3) HAS_COMMA(PASTE5(_IS_EMPTY_CASE_, _0, _1, _2, _3))
// define only one _IS_EMPTY_CASE to contain a comma, so _ISEMPTY(...) evaluates to 1 only for it
#define _IS_EMPTY_CASE_0001 ,

// helper to trigger a comma only when invoked with parenthesis
#define _TRIGGER_PARENTHESIS_(...) ,

// main macro feeds _IS_EMPTY with test cases
#define BS_ISEMPTY(...)                                                 \
_ISEMPTY(                                                               \
          /* test if there is just one argument, eventually an empty    \
             one */                                                     \
          HAS_COMMA(__VA_ARGS__),                                       \
          /* test if _TRIGGER_PARENTHESIS_ together with the argument   \
             adds a comma */                                            \
          HAS_COMMA(_TRIGGER_PARENTHESIS_ __VA_ARGS__),                 \
          /* test if the argument together with a parenthesis           \
             adds a comma */                                            \
          HAS_COMMA(__VA_ARGS__ (/*empty*/)),                           \
          /* test if placing it between _TRIGGER_PARENTHESIS_ and the   \
             parenthesis adds a comma */                                \
          HAS_COMMA(_TRIGGER_PARENTHESIS_ __VA_ARGS__ (/*empty*/))      \
          )
// test tuples for emptiness
#define BS_IS_EMPTY_TUPLE(tup) BS_ISEMPTY(BOOST_PP_TUPLE_ENUM(tup))
// test if tuple is non-empty
#define BS_IS_NONEMPTY_TUPLE(tup) BOOST_PP_NOT(BS_IS_EMPTY_TUPLE(tup))

// macro that returns 0 for tuple `()` with empty element (boost macro returns 1)
#define BS_TUPLE_SIZE(tup) \
BOOST_PP_IIF(BS_ISEMPTY(BOOST_PP_TUPLE_ENUM(tup)), 0, BOOST_PP_SIZE(tup))

//----
// expands to: < A0, A1, ..., An-1 >, if n > 0
// otherwise to nothing
#define BS_ENUM_TPL_ARGS_0(...)
#define BS_ENUM_TPL_ARGS_1(args_num) < BOOST_PP_ENUM_PARAMS(args_num, A) >
// expands to: T< A0, A1, ..., An-1 >, if n > 0
// otherwise to T
#define BS_ENUM_TPL(T, args_num) \
T BOOST_PP_CAT(BS_ENUM_TPL_ARGS_, BOOST_PP_BOOL(args_num))(args_num)

//----
// expands to: < tpl_args[0], tpl_args[0], ..., tpl_args[n - 1] >, braces present if `braces` == 1
#define BS_UNFOLD_TPL_ARGS_0(...)
#define BS_UNFOLD_TPL_ARGS_1(tpl_args, braces) \
BOOST_PP_EXPR_IIF(braces, <) BOOST_PP_TUPLE_ENUM(tpl_args) BOOST_PP_EXPR_IIF(braces, >)
// expands to: tpl_args[0], tpl_args[0], ..., tpl_args[n - 1], if n > 0
// otherwise to nothing
// `tpl_args` is a tuple!
#define BS_UNFOLD_TPL_ARGS(T, tpl_args) \
BOOST_PP_CAT(BS_UNFOLD_TPL_ARGS_, BS_IS_NONEMPTY_TUPLE(tpl_args))(tpl_args, 0)
// expands to: T< tpl_args[0], tpl_args[0], ..., tpl_args[n - 1] >, if n > 0
// otherwise to T
#define BS_UNFOLD_TPL_T(T, tpl_args) \
T BOOST_PP_CAT(BS_UNFOLD_TPL_ARGS_, BS_IS_NONEMPTY_TUPLE(tpl_args))(tpl_args, 1)

//----
// expands to: < args_prefix[0] A0, args_prefix[1] A1, ..., args_prefix[n-1] An-1 >, braces if `braces` == 1
// `args_prefix` is a tuple!
#define BS_UNFOLD_TPL_PREFIX_0(...)
#define BS_UNFOLD_TPL_PREFIX_1(args_prefix, braces) \
BOOST_PP_EXPR_IIF(braces, <) BOOST_PP_SEQ_FOR_EACH_I(BS_TPL_ARG_, A, BOOST_PP_TUPLE_TO_SEQ(args_prefix)) BOOST_PP_EXPR_IIF(braces, >)
// expands to: args_prefix[0] A0, args_prefix[1] A1, ..., args_prefix[n-1] An-1, if n > 0
// otherwise to nothing
#define BS_UNFOLD_TPL_PREFIX(args_prefix) \
BOOST_PP_CAT(BS_UNFOLD_TPL_PREFIX_, BS_IS_NONEMPTY_TUPLE(args_prefix))(args_prefix, 0)
// expands to: T < args_prefix[0] A0, args_prefix[1] A1, ..., args_prefix[n-1] An-1 >, if n > 0
// otherwise to T
#define BS_UNFOLD_TPL_PREFIX_T(T, args_prefix) \
T BOOST_PP_CAT(BS_UNFOLD_TPL_PREFIX_, BS_IS_NONEMPTY_TUPLE(args_prefix))(args_prefix, 1)

/*-----------------------------------------------------------------------------
 *  register polymorphic type in cereal using type name from `type_descriptor`
 *-----------------------------------------------------------------------------*/
#define BSS_REGISTER_TYPE(...)                        \
namespace cereal { namespace detail {                 \
template <>                                           \
struct binding_name<__VA_ARGS__>  {                   \
   static char const * name() {                       \
        return __VA_ARGS__::bs_type().name.c_str(); } \
};                                                    \
} } /* end namespaces */                              \
CEREAL_BIND_TO_ARCHIVES(__VA_ARGS__)

/*-----------------------------------------------------------------------------
 *  helpers to generate signatures of `blue_sky::atomizer::[serialization function]::go()`
 *-----------------------------------------------------------------------------*/
#define BSS_FCN_save(Archive, T, tpl_args_prefix) go( \
    Archive& ar, BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) const& t, std::uint32_t const version \
) -> void

#define BSS_FCN_load(Archive, T, tpl_args_prefix) go( \
    Archive& ar, BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix))& t, std::uint32_t const version \
) -> void

#define BSS_FCN_serialize(Archive, T, tpl_args_prefix) go( \
    Archive& ar, BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix))& t, std::uint32_t const version \
) -> void

#define BSS_FCN_save_minimal(Archive, T, tpl_args_prefix) go( \
    Archive const& ar, BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) const& t, std::uint32_t const version)

#define BSS_FCN_load_minimal(Archive, T, tpl_args_prefix) go( \
    Archive const& ar, BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix))& t, \
    V const& v, std::uint32_t const version)

#define BSS_FCN_load_and_construct(Archive, T, tpl_args_prefix) go( \
    Archive& ar, cereal::construct< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) >& c, \
    std::uint32_t const version \
) -> void

/*-----------------------------------------------------------------------------
 *  declare `blue_sky::atomizer::[serialization function]` (serialize, save/load, ...)
 *-----------------------------------------------------------------------------*/
// generate overload of cereal::LoadAndConstruct struct
#define BSS_CEREAL_OVERLOAD_load_and_construct(T, tpl_args_prefix)                                              \
namespace cereal {                                                                                              \
    template< BS_UNFOLD_TPL_PREFIX(tpl_args_prefix) >                                                           \
    struct LoadAndConstruct< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) > {                                 \
        template<typename Archive>                                                                              \
        static auto load_and_construct(Archive& ar,                                                             \
            cereal::construct< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) >& c, std::uint32_t const version \
        ) -> void { ::blue_sky::atomizer::load_and_construct(ar, c, version); }                                 \
    };                                                                                                          \
}

// detect that passed function is `load_minimal`
#define IS_FCN_load_minimal(fcn) HAS_COMMA(BOOST_PP_CAT(_IS1_FCN_, fcn))
#define _IS1_FCN_load_minimal ,
// detect that passed function is `load_and_construct`
#define IS_FCN_load_and_construct(fcn) HAS_COMMA(BOOST_PP_CAT(_IS2_FCN_, fcn))
#define _IS2_FCN_load_and_construct ,

// extended version of main DECL macro
// tpl_args_prefix - tuple that specifies template prefixes, can be empty ()
#define BSS_FCN_DECL_EXT(fcn, T, tpl_args_prefix)                                  \
BOOST_PP_EXPR_IIF(IS_FCN_load_and_construct(fcn), BSS_CEREAL_OVERLOAD_load_and_construct(T, tpl_args_prefix)) \
template< BS_UNFOLD_TPL_PREFIX(tpl_args_prefix) >                                  \
struct blue_sky::atomizer::fcn< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) > { \
    BOOST_PP_TUPLE_ENUM(BOOST_PP_IIF(IS_FCN_load_minimal(fcn),                     \
        (template<typename Archive, typename V>),                                  \
        (template<typename Archive>)                                               \
    )) static auto BSS_FCN_##fcn(Archive, T, tpl_args_prefix); };

// simpler version when only template args number is specified or type isn't a template
#define BSS_FCN_DECL_T(fcn, T, tpl_args_num) \
BSS_FCN_DECL_EXT(fcn, T, (BOOST_PP_ENUM(tpl_args_num, BS_ECHO_, typename)))

#define BSS_FCN_DECL(fcn, T) \
BSS_FCN_DECL_EXT(fcn, T, ())

/*-----------------------------------------------------------------------------
 *  define begin/end of BS serialization structs
 *-----------------------------------------------------------------------------*/
// [NOTE] automatically adds `type` alias equal to fully qualified `T`
#define BSS_FCN_BEGIN_EXT(fcn, T, tpl_args_prefix)                               \
BOOST_PP_TUPLE_ENUM(BOOST_PP_IIF(IS_FCN_load_minimal(fcn),                       \
    (template<typename Archive, typename V>),                                    \
    (template<typename Archive>) ))                                              \
auto ::blue_sky::atomizer::fcn< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) > \
::BSS_FCN_##fcn(Archive, T, tpl_args_prefix) {                                   \
    using type = BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix));

// simpler version when only template args number is specified or type isn't a template
#define BSS_FCN_BEGIN_T(fcn, T, tpl_args_num) \
BSS_FCN_BEGIN_EXT(fcn, T, (BOOST_PP_ENUM(tpl_args_num, BS_ECHO_, typename)))

#define BSS_FCN_BEGIN(fcn, T) \
BSS_FCN_BEGIN_EXT(fcn, T, ())

#define BSS_FCN_END \
(void)ar; (void)t; (void)version; }

/*-----------------------------------------------------------------------------
 *  generate explicit specializations of `blue_sky::atomizer::[serialization function]`
 *  for commonly used archives and mark 'em as exported
 *-----------------------------------------------------------------------------*/
#define BSS_FCN_EXPORT_EXT(fcn, T, tpl_args_prefix)                                                         \
template<> BS_API_PLUGIN auto ::blue_sky::atomizer::fcn< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) >:: \
BSS_FCN_##fcn(::cereal::JSONInputArchive, T, tpl_args_prefix);                                              \
template<> BS_API_PLUGIN auto ::blue_sky::atomizer::fcn< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) >:: \
BSS_FCN_##fcn(::cereal::JSONOutputArchive, T, tpl_args_prefix);                                             \
template<> BS_API_PLUGIN auto ::blue_sky::atomizer::fcn< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) >:: \
BSS_FCN_##fcn(::cereal::BinaryInputArchive, T, tpl_args_prefix);                                            \
template<> BS_API_PLUGIN auto ::blue_sky::atomizer::fcn< BS_ENUM_TPL(T, BS_TUPLE_SIZE(tpl_args_prefix)) >:: \
BSS_FCN_##fcn(::cereal::BinaryOutputArchive, T, tpl_args_prefix);                                           \

// simpler version when only template args number is specified or type isn't a template
#define BSS_FCN_EXPORT_T(fcn, T, tpl_args_num) \
BSS_FCN_EXPORT_EXT(fcn, T, (BOOST_PP_ENUM(tpl_args_num, BS_ECHO_, typename)))

#define BSS_FCN_EXPORT(fcn, T) \
BSS_FCN_EXPORT_EXT(fcn, T, ())

