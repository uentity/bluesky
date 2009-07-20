/**
 * \file pp_param_list.h
 * \brief converts sequence to list of parameters
 * \author Sergey Miryanov
 * \date 24.07.2008
 * */
#ifndef BS_PP_PARAM_LIST_H_
#define BS_PP_PARAM_LIST_H_

#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#define BS_PP_TUPLE_ELEM(tuple_, size_, rsize_) \
  BOOST_PP_TUPLE_ELEM (size_, BOOST_PP_SUB (size_, rsize_), tuple_)

#define BS_PP_ELEM_NAME(name_, size_, rsize_) \
  BOOST_PP_CAT (name_, BOOST_PP_SUB (size_, rsize_))

/** 
 * */
#define SIMPLE_TYPE_LIST(size_, tuple_) \
  BOOST_PP_SEQ_ENUM (BOOST_PP_TUPLE_TO_SEQ (size_, tuple_))

/** 
 * */
#define TYPE_LIST_DECL(tuple_, size_) BOOST_PP_CAT (TYPE_LIST_DECL_, size_) (tuple_, size_, size_)
#define TYPE_LIST_DECL_0(tuple_, size_, rsize_)
#define TYPE_LIST_DECL_1(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_)
#define TYPE_LIST_DECL_2(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_1  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_3(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_2  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_4(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_3  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_5(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_4  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_6(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_5  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_7(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_6  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_8(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_7  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_9(tuple_, size_, rsize_)  typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_8  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_10(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_9  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_11(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_10 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_12(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_11 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_13(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_12 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_14(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_13 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_15(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_14 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_16(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_15 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_17(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_16 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_DECL_18(tuple_, size_, rsize_) typename BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_DECL_17 (tuple_, size_, BOOST_PP_DEC (rsize_))

/** 
 * */
#define TYPE_LIST(tuple_, size_) BOOST_PP_CAT (TYPE_LIST_, size_) (tuple_, size_, size_)
#define TYPE_LIST_0(tuple_, size_, rsize_)
#define TYPE_LIST_1(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_)
#define TYPE_LIST_2(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_1  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_3(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_2  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_4(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_3  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_5(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_4  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_6(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_5  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_7(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_6  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_8(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_7  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_9(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_8  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_10(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_9  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_11(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_10 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_12(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_11 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_13(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_12 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_14(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_13 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_15(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_14 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_16(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_15 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_17(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_16 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define TYPE_LIST_18(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_), TYPE_LIST_17 (tuple_, size_, BOOST_PP_DEC (rsize_))

/** 
 * */
#define PARAM_LIST(tuple_, size_) BOOST_PP_CAT (PARAM_LIST_, size_) (tuple_, size_, size_)
#define PARAM_LIST_0(tuple_, size_, rsize_)
#define PARAM_LIST_1(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_)
#define PARAM_LIST_2(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_1  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_3(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_2  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_4(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_3  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_5(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_4  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_6(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_5  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_7(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_6  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_8(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_7  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_9(tuple_, size_, rsize_)  BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_8  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_10(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_9  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_11(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_10 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_12(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_11 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_13(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_12 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_14(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_13 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_15(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_14 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_16(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_15 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_17(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_16 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define PARAM_LIST_18(tuple_, size_, rsize_) BS_PP_TUPLE_ELEM (tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), PARAM_LIST_17 (tuple_, size_, BOOST_PP_DEC (rsize_))

/**
 * */
#define PARAM_LIST_FOR_EACH(tuple_, size_)  \
  PARAM_LIST_FOR_EACH_INTERNAL (BOOST_PP_TUPLE_TO_SEQ (size_, tuple_), size_)

#define PARAM_LIST_FOR_EACH_INTERNAL(seq_, len_)  BOOST_PP_CAT (PARAM_LIST_FOR_EACH_INTERNAL_, len_) seq_
#define PARAM_LIST_FOR_EACH_INTERNAL_0
#define PARAM_LIST_FOR_EACH_INTERNAL_1(x)  x A1
#define PARAM_LIST_FOR_EACH_INTERNAL_2(x)  x A2,  PARAM_LIST_FOR_EACH_INTERNAL_1
#define PARAM_LIST_FOR_EACH_INTERNAL_3(x)  x A3,  PARAM_LIST_FOR_EACH_INTERNAL_2
#define PARAM_LIST_FOR_EACH_INTERNAL_4(x)  x A4,  PARAM_LIST_FOR_EACH_INTERNAL_3
#define PARAM_LIST_FOR_EACH_INTERNAL_5(x)  x A5,  PARAM_LIST_FOR_EACH_INTERNAL_4
#define PARAM_LIST_FOR_EACH_INTERNAL_6(x)  x A6,  PARAM_LIST_FOR_EACH_INTERNAL_5
#define PARAM_LIST_FOR_EACH_INTERNAL_7(x)  x A7,  PARAM_LIST_FOR_EACH_INTERNAL_6
#define PARAM_LIST_FOR_EACH_INTERNAL_8(x)  x A8,  PARAM_LIST_FOR_EACH_INTERNAL_7
#define PARAM_LIST_FOR_EACH_INTERNAL_9(x)  x A9,  PARAM_LIST_FOR_EACH_INTERNAL_8
#define PARAM_LIST_FOR_EACH_INTERNAL_10(x) x A10, PARAM_LIST_FOR_EACH_INTERNAL_9
#define PARAM_LIST_FOR_EACH_INTERNAL_11(x) x A11, PARAM_LIST_FOR_EACH_INTERNAL_10
#define PARAM_LIST_FOR_EACH_INTERNAL_12(x) x A12, PARAM_LIST_FOR_EACH_INTERNAL_11
#define PARAM_LIST_FOR_EACH_INTERNAL_13(x) x A13, PARAM_LIST_FOR_EACH_INTERNAL_12
#define PARAM_LIST_FOR_EACH_INTERNAL_14(x) x A14, PARAM_LIST_FOR_EACH_INTERNAL_13
#define PARAM_LIST_FOR_EACH_INTERNAL_15(x) x A15, PARAM_LIST_FOR_EACH_INTERNAL_14
#define PARAM_LIST_FOR_EACH_INTERNAL_16(x) x A16, PARAM_LIST_FOR_EACH_INTERNAL_15
#define PARAM_LIST_FOR_EACH_INTERNAL_17(x) x A17, PARAM_LIST_FOR_EACH_INTERNAL_16
#define PARAM_LIST_FOR_EACH_INTERNAL_18(x) x A18, PARAM_LIST_FOR_EACH_INTERNAL_17

/**
 * */
#define CALL_LIST(tuple_, size_) BOOST_PP_CAT (CALL_LIST_, size_) (tuple_, size_, size_)
#define CALL_LIST_0(tuple_, size_, rsize_)
#define CALL_LIST_1(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_)
#define CALL_LIST_2(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_1  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_3(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_2  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_4(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_3  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_5(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_4  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_6(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_5  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_7(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_6  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_8(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_7  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_9(tuple_, size_, rsize_)  BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_8  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_10(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_9  (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_11(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_10 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_12(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_11 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_13(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_12 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_14(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_13 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_15(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_14 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_16(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_15 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_17(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_16 (tuple_, size_, BOOST_PP_DEC (rsize_))
#define CALL_LIST_18(tuple_, size_, rsize_) BS_PP_ELEM_NAME (A, size_, rsize_), CALL_LIST_17 (tuple_, size_, BOOST_PP_DEC (rsize_))

namespace blue_sky {
  enum {
    empty_arg__ = 0,
  };
}

#endif	// #ifndef BS_PP_PARAM_LIST_H_
