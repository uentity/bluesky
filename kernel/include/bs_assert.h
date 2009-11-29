/**
 * \file bs_assert.h
 * \brief smart assertation, based on Alexandrescu's ideas
 * \author Sergey Miryanov
 * \date 14.05.2008
 */
#ifndef BS_ASSERT_H_
#define BS_ASSERT_H_

#include "bs_common.h"
#include <boost/lexical_cast.hpp>

namespace blue_sky {

  namespace detail {

    inline std::string
    get_str (void *const p)
    {
      char tmp[1024] = {0};
      size_t len = sprintf (tmp, "%p", p);

      return std::string (tmp, len);
    }

    inline std::string
    get_str (const size_t &t)
    {
      char tmp[1024] = {0};
      size_t len = sprintf (tmp, "%lu", t);

      return std::string (tmp, len);
    }

    inline std::string
    get_str (const int &t)
    {
      char tmp[1024] = {0};
      size_t len = sprintf (tmp, "%d", t);

      return std::string (tmp, len);
    }

#ifndef UNIX
    inline std::string
    get_str (const unsigned long int &t)
    {
      char tmp[1024] = {0};
      size_t len = sprintf (tmp, "%lu", t);

      return std::string (tmp, len);
    }
#endif

    inline std::string
    get_str (const double &t)
    {
      return boost::lexical_cast <std::string> (t);
    }

    inline std::string
    get_str (const char *t)
    {
      return t;
    }

    inline std::string
    get_str (const std::string &t)
    {
      return t;
    }

    template <typename T>
    inline std::string
    get_str (const T &t)
    {
      return boost::lexical_cast <std::string> (t);
    }
  }

  namespace bs_assert {

  //! forward declaration
  class assert_factory;

  /**
   * \brief perform assert actions
   * */
  class BS_API_PLUGIN asserter
  {
  public:

	  /**
	   * \brief user reaction on assert
	   * */
	  enum assert_state
	  {
		  STATE_KILL,       // ! terminate process
		  STATE_BREAK,      // ! break into file where assertion failed
		  STATE_IGNORE,     // ! ignore current assert
		  STATE_IGNORE_ALL, // ! ignore current and all following asserts
	  };

  public:

	  asserter (const char *file, int line, const char *cond_str)
		  : cond (true)
		    , file (file)
		    , line (line)
		    , cond_s (cond_str)
		    , var_list ("")
	  {
		  ASSERTER_A = this;
		  ASSERTER_B = this;
	  }

	  asserter (bool cond, const char *file, int line, const char *cond_str)
		  : cond (cond)
		    , file (file)
		    , line (line)
		    , cond_s (cond_str)
		    , var_list ("")
	  {
		  ASSERTER_A = this;
		  ASSERTER_B = this;
	  }

	  virtual ~asserter ()
	  {
	  }

	  /**
	   * \brief ask user what he want to do in case of false condition
	   * */
	  virtual assert_state ask_user () const;
	  /**
	   * \brief perform action that user select in ask_user
	   * */
	  virtual bool handle () const;

	  /**
	   * \brief break program execution (through int 3, for example)
	   * */
	  void break_here () const;

	  /**
	   * \brief pass filename, line no and string condition to asserter and
	   *		  return asserter object. asserter object will be used to create
	   *		  real asserter through call to asserter::make function.
	   * */
	  static asserter
	  workaround (const char * file_, int line_, const char *cond_str)
	  {
		  return asserter (file_, line_, cond_str);
	  }

	  /**
	   * \brief create real asserter through call to factory ()->make function
	   *		  may return asserter that defined in python
	   * */
	  asserter *make (bool cond);

	  /**
	   * \brief add variable and it value to variable list
	   *		  variable list will be printed in case of false condition
	   * */
	  template <class T> asserter *
	  add_var (const T &t, const std::string &name)
	  {
      std::string str = name + " = " + detail::get_str (t) + "\n";
		  var_list = var_list + str;

		  return this;
	  }

	  /**
	   * \brief ignore all following asserts or no
	   * */
	  inline bool &
	  ignore_all () const
	  {
		  static bool ignore_all_ = false;
		  return ignore_all_;
	  }

	  static void
	  set_factory (assert_factory *f)
	  {
		  factory() = f;
	  }

	  static assert_factory *&
	  factory ()
	  {
		  static assert_factory *factory_ = 0;
		  return factory_;
	  }

  public:

	  /** 
	   * brief helpers to call add_var
	   * */
	  asserter *ASSERTER_A;
	  asserter *ASSERTER_B;

  public:
	  bool                   cond;
	  const char            *file;
	  int                    line;
	  const char            *cond_s;
	  std::string var_list;
  };

  /**
   * \brief assert factory used to create an objects of assert class (or any in assert
   *		hierarchy, i.e. assert_factory inherited in python and create python version 
   *		of asserter)
   * */
  class BS_API_PLUGIN assert_factory
  {
  public:
	  virtual ~assert_factory () {}

	  virtual asserter *make (bool b, const char *file, int line, const char *cond_str);
  };

  /** 
   * \brief helper class to check asserter condition, 
   *		handle assert if cond is false and break if needed
   * */
  struct BS_API_PLUGIN assert_wrapper
  {
	  assert_wrapper (asserter *pa);
  };


//! define macro to check condition in debug version
#ifdef _DEBUG
#define BS_ASSERT(cond) \
  if (false) ; else     \
  bs_assert::assert_wrapper wrapper_ = bs_assert::asserter::workaround(__FILE__, __LINE__, (#cond)).make(!!(cond))->ASSERTER_A
#else
#define BS_ASSERT(cond) \
  if (true) ; else      \
  bs_assert::assert_wrapper wrapper_ = bs_assert::asserter::workaround(__FILE__, __LINE__, (#cond)).make(!!(cond))->ASSERTER_A
#endif

#ifdef _DEBUG
#define BS_ERROR(cond, caption)          \
  if (false) ; else {                    \
  BS_ASSERT ((cond));                    \
  if (!(cond))                           \
    throw bs_exception (caption, #cond); \
  }
#else
#define BS_ERROR(cond, caption)          \
  if (!(cond)) {                         \
    throw bs_exception (caption, #cond); \
  }
#endif

#define ASSERTER_A(x)			ASSERTER_OP_(x, B)
#define ASSERTER_B(x)			ASSERTER_OP_(x, A)
#define ASSERTER_OP_(x, next)	ASSERTER_A->add_var ((x), #x)->ASSERTER_##next

  } // namespace bs_assert

} // namespace blue_sky


#endif	// #ifndef BS_ASSERT_H_

