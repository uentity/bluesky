/**
 * */
#ifndef BLUE_SKY_KERNEL_SIGNALS_H_
#define BLUE_SKY_KERNEL_SIGNALS_H_

#include "pp_param_list.h"
#include <boost/preprocessor/cat.hpp>
#include <boost/function.hpp>
#include <boost/signal.hpp>
#include <boost/bind.hpp>

namespace blue_sky {

#define DECL_RAISE_I(name, p, ps)                                                   \
  void                                                                              \
  BOOST_PP_CAT (on_, name) (PARAM_LIST (p, ps)) const                               \
  {                                                                                 \
    BOOST_PP_CAT (name, _signal_)->operator() (CALL_LIST (p, ps));                  \
  }

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

#define DECL_SIGNAL_I(name, p, ps)                                                  \
  typedef boost::function <void (TYPE_LIST (p, ps))>                                \
    BOOST_PP_CAT (name, _functor_t);                                                \
  typedef boost::signal <void (TYPE_LIST (p, ps))> BOOST_PP_CAT (name, _signal_t);  \
  typedef boost::shared_ptr <BOOST_PP_CAT (name, _signal_t)>                        \
    BOOST_PP_CAT (name, _sp_signal_t);                                              \
  BOOST_PP_CAT (name, _sp_signal_t) BOOST_PP_CAT (name, _signal_);

#define DECL_INIT_SIGNAL_I(name, p, ps)                                             \
  BOOST_PP_CAT (name, _signal_).reset (new BOOST_PP_CAT (name, _signal_t));

#define DECL_DISCONNECT_SIGNAL_I(name, p, ps)                                       \
  BOOST_PP_CAT (name, _signal_).reset ();

#define RAISE_I(r, data, i, elem)                                                   \
  DECL_RAISE_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                                   \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

#define ADD_I(r, data, i, elem)                                                     \
  DECL_ADD_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                                     \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

#define SIGNALS_I(r, data, i, elem)                                                 \
  DECL_SIGNAL_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                                  \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

#define INIT_SIGNAL_I(r, data, i, elem)                                             \
  DECL_INIT_SIGNAL_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                             \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

#define DISCONNECT_SIGNAL_I(r, data, i, elem)                                       \
  DECL_DISCONNECT_SIGNAL_I (BOOST_PP_TUPLE_ELEM (3, 0, elem),                       \
    BOOST_PP_TUPLE_ELEM (3, 1, elem),                                               \
    BOOST_PP_TUPLE_ELEM (3, 2, elem))

#define DECLARE_EVENT_LIST_V2(owner, seq)                                           \
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

  struct signals_disconnector
  {
    virtual ~signals_disconnector () {}

    virtual void
    disconnect_signals () = 0;
  };

} // namespace blue_sky




#endif // #ifndef BLUE_SKY_KERNEL_SIGNALS_H_

