/**
 *       \file  kernel_signals.h
 *      \brief  New version of synchronious signal system, for each
 *              signal allocated boost::signal with 'right' type and
 *              signature
 *     \author  Sergey Miryanov (sergey-miryanov), sergey.miryanov@gmail.com
 *       \date  16.12.2009
 *  \copyright  This source code is released under the terms of 
 *              the BSD License. See LICENSE for more details.
 * */
#ifndef BLUE_SKY_KERNEL_SIGNALS_H_
#define BLUE_SKY_KERNEL_SIGNALS_H_

#include "pp_param_list.h"
#include <boost/preprocessor/cat.hpp>
#include <boost/function.hpp>
#include <boost/signal.hpp>
#include <boost/bind.hpp>

// Don't include any Boost.Python header in this file

namespace blue_sky {

/**
 * \brief  Declares function that raised (fired) signal
 * \param  name Signal name
 * \param  p Tuple with signal parameters
 * \param  ps Tuple size
 * */
#define DECL_RAISE_I(name, p, ps)                                                   \
  void                                                                              \
  BOOST_PP_CAT (on_, name) (PARAM_LIST (p, ps)) const                               \
  {                                                                                 \
    BOOST_PP_CAT (name, _signal_)->operator() (CALL_LIST (p, ps));                  \
  }

/**
 * \brief  Declares functions that added slots to signal
 * \param  name Signal name
 * \param  p Tuple with signal parameters
 * \param  ps Tuple size
 * */
#define DECL_ADD_I(name, p, ps)                                                     \
  void                                                                              \
  BOOST_PP_CAT (BOOST_PP_CAT (add_, name), _handler) (                              \
    const BOOST_PP_CAT (name, _functor_t) &handler)                                 \
  {                                                                                 \
    BOOST_PP_CAT (name, _signal_)->connect (handler);                               \
  }                                                                                 \
  template <typename BOOST_PP_CAT (name, _T), typename P>                           \
  void                                                                              \
  BOOST_PP_CAT (BOOST_PP_CAT (add_, name), _handler) (                              \
    void (BOOST_PP_CAT (name, _T)::*handler) (TYPE_LIST (p, ps)),                   \
    const P &t                                                                      \
    )                                                                               \
  {                                                                                 \
    BOOST_PP_CAT (name, _signal_)->connect (                                        \
      boost::bind (std::mem_fun (handler), t, _1));                                 \
  }                                                                                 \
  template <typename BOOST_PP_CAT (name, _T), typename P>                           \
  void                                                                              \
  BOOST_PP_CAT (BOOST_PP_CAT (add_, name), _handler) (                              \
    void (BOOST_PP_CAT (name, _T)::*handler) (TYPE_LIST (p, ps)) const,             \
    const P &t                                                                      \
    )                                                                               \
  {                                                                                 \
    BOOST_PP_CAT (name, _signal_)->connect (                                        \
      boost::bind (std::mem_fun (handler), t, _1));                                 \
  }

/**
 * \brief  Declares function that added slots (from python) to signal
 * \param  name Signal name
 * \param  p Tuple with signal parameters
 * \param  ps Tuple size
 * */
#define DECL_PY_ADD_I(name, p, ps)                                                  \
  void                                                                              \
  BOOST_PP_CAT (BOOST_PP_CAT (py_add_, name), _handler) (                           \
    const boost::python::object &handler)                                           \
  {                                                                                 \
    BOOST_PP_CAT (name, _signal_)->connect (handler);                               \
  }

/**
 * \brief  Declares signal and signal types (boost::signal and boost::function)
 * \param  name Signal name
 * \param  p Tuple with signal parameters
 * \param  ps Tuple size
 * */
#define DECL_SIGNAL_I(name, p, ps)                                                  \
  typedef boost::function <void (TYPE_LIST (p, ps))>                                \
    BOOST_PP_CAT (name, _functor_t);                                                \
  typedef boost::signal <void (TYPE_LIST (p, ps))> BOOST_PP_CAT (name, _signal_t);  \
  typedef boost::shared_ptr <BOOST_PP_CAT (name, _signal_t)>                        \
    BOOST_PP_CAT (name, _sp_signal_t);                                              \
  BOOST_PP_CAT (name, _sp_signal_t) BOOST_PP_CAT (name, _signal_);

/**
 * \brief  Declares signal initialization
 * \param  name Signal name
 * \param  p Tuple with signal parameters
 * \param  ps Tuple size
 * */
#define DECL_INIT_SIGNAL_I(name, p, ps)                                             \
  BOOST_PP_CAT (name, _signal_).reset (new BOOST_PP_CAT (name, _signal_t));

/**
 * \brief  Declares signal disconnection
 * \param  name Signal name
 * \param  p Tuple with signal parameters
 * \param  ps Tuple size
 * */
#define DECL_DISCONNECT_SIGNAL_I(name, p, ps)                                       \
  BOOST_PP_CAT (name, _signal_).reset ();

/**
 * \brief  Declares export of 'add' function to python
 * \param  owner Class which owns this signal
 * \param  name Signal name
 * \param  p Tuple with signal parameters
 * \param  ps Tuple size
 * */
#define DECL_PY_EXPORT_I(owner, name, p, ps)                                        \
  class__.def (                                                                     \
    BOOST_PP_STRINGIZE (BOOST_PP_CAT (BOOST_PP_CAT (add_, name), _handler)),        \
    &owner::BOOST_PP_CAT (BOOST_PP_CAT (py_add_, name), _handler));

/**
 * \brief  Declares list of 'raise' functions
 * \param  See Boost.Preprocessor/BOOST_PP_SEQ_FOR_EACH_I
 * */
#define RAISE_I(r, data, i, elem)                                                   \
  DECL_RAISE_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                                   \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

/**
 * \brief  Declares list of 'add' functions
 * \param  See Boost.Preprocessor/BOOST_PP_SEQ_FOR_EACH_I
 * */
#define ADD_I(r, data, i, elem)                                                     \
  DECL_ADD_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                                     \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

/**
 * \brief  Declares list of python versions of 'add' functions
 * \param  See Boost.Preprocessor/BOOST_PP_SEQ_FOR_EACH_I
 * */
#define PY_ADD_I(r, data, i, elem)                                                  \
  DECL_PY_ADD_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                                  \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

/**
 * \brief  Declares list of signals
 * \param  See Boost.Preprocessor/BOOST_PP_SEQ_FOR_EACH_I
 * */
#define SIGNALS_I(r, data, i, elem)                                                 \
  DECL_SIGNAL_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                                  \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

/**
 * \brief  Declares initialization of signal list
 * \param  See Boost.Preprocessor/BOOST_PP_SEQ_FOR_EACH_I
 * */
#define INIT_SIGNAL_I(r, data, i, elem)                                             \
  DECL_INIT_SIGNAL_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                             \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

/**
 * \brief  Declares disconnection of signal list
 * \param  See Boost.Preprocessor/BOOST_PP_SEQ_FOR_EACH_I
 * */
#define DISCONNECT_SIGNAL_I(r, data, i, elem)                                       \
  DECL_DISCONNECT_SIGNAL_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                       \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

/**
 * \brief  Declares export of 'add' functions to python
 * \param  See Boost.Preprocessor/BOOST_PP_SEQ_FOR_EACH_I
 * */
#define PY_EXPORT_I(r, data, i, elem)                                               \
  DECL_PY_EXPORT_I (data, BOOST_PP_TUPLE_ELEM (3, 0, elem),                         \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

/**
 * \brief  Declares event list and util function and classes
 * \param  owner Name of event list owner
 * \param  seq List of signals (each 'signal' is a tuple 
 *             (signal name, params, params number)
 * */
#define DECLARE_EVENT_LIST_V2_IMPL(owner, seq)                                      \
  BOOST_PP_SEQ_FOR_EACH_I (RAISE_I, _, seq)                                         \
  BOOST_PP_SEQ_FOR_EACH_I (SIGNALS_I, _, seq)                                       \
  BOOST_PP_SEQ_FOR_EACH_I (ADD_I, _, seq)                                           \
  void                                                                              \
  BOOST_PP_CAT (owner, _private_init_signals) ()                                    \
  {                                                                                 \
    BOOST_PP_SEQ_FOR_EACH_I (INIT_SIGNAL_I, _, seq)                                 \
  }                                                                                 \
  void                                                                              \
  BOOST_PP_CAT (owner, _private_disconnect_signals) ()                              \
  {                                                                                 \
    BOOST_PP_SEQ_FOR_EACH_I (DISCONNECT_SIGNAL_I, _, seq)                           \
  }                                                                                 \
  struct BOOST_PP_CAT (owner, _events_init) : signals_disconnector                  \
  {                                                                                 \
    BOOST_PP_CAT (owner, _events_init) (owner *owner_)                              \
    : owner_ (owner_)                                                               \
    {                                                                               \
      owner_->BOOST_PP_CAT (owner, _private_init_signals) ();                       \
      BS_KERNEL.register_disconnector (this);                                       \
    }                                                                               \
    ~BOOST_PP_CAT (owner, _events_init) ()                                          \
    {                                                                               \
      BS_KERNEL.unregister_disconnector (this);                                     \
    }                                                                               \
    virtual void                                                                    \
    disconnect_signals ()                                                           \
    {                                                                               \
      owner_->BOOST_PP_CAT (owner, _private_disconnect_signals) ();                 \
    }                                                                               \
    owner *owner_;                                                                  \
  };                                                                                \
  BOOST_PP_CAT (owner, _events_init) BOOST_PP_CAT (owner, _events_init_);

#ifdef BSPY_EXPORTING_PLUGIN
#define DECLARE_EVENT_LIST_V2(owner, seq)                                           \
  DECLARE_EVENT_LIST_V2_IMPL (owner, seq)                                           \
  BOOST_PP_SEQ_FOR_EACH_I (PY_ADD_I, _, seq)                                        \
  template <typename class_t>                                                       \
  static class_t &                                                                  \
  python_exporter (class_t &class__)                                                \
  {                                                                                 \
    BOOST_PP_SEQ_FOR_EACH_I (PY_EXPORT_I, owner, seq)                               \
    return base_t::python_exporter (class__);                                       \
  }
#else
#define DECLARE_EVENT_LIST_V2(owner, seq)                                           \
  DECLARE_EVENT_LIST_V2_IMPL (owner, seq)
#endif


  /**
   * \class signals_disconnector
   * \brief Disconnects all signals for owner (holded in child class)
   * */
  struct signals_disconnector
  {
    virtual ~signals_disconnector () {}

    /**
     * \brief  Disconnects all signals for owner
     * */
    virtual void
    disconnect_signals () = 0;
  };

} // namespace blue_sky




#endif // #ifndef BLUE_SKY_KERNEL_SIGNALS_H_

