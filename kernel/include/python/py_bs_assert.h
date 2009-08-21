/**
 * \file py_bs_assert.h
 * \brief python wrapper for bs_assert
 * \author Sergey Miryanov
 * \date 14.05.2008
 * */
#ifndef PY_BS_ASSERT_H_
#define PY_BS_ASSERT_H_

#include "bs_assert.h"
#include <boost/python.hpp>

namespace blue_sky {
namespace python {

//! forward declaration
class py_assert_factory;

/** 
 * \brief class helper to set assert factory from python 
 * */
class py_assert_factory_setter
{
public:

	py_assert_factory_setter () {}

	void set_factory (py_assert_factory *factory);
};

/** 
 * \brief assert factory - for inheritance from python
 * */
class py_assert_factory : public bs_assert::assert_factory, public boost::python::wrapper<bs_assert::assert_factory>
{
public:

	py_assert_factory ()
		: disable_python_call_on_success (true)
	{}

	virtual ~py_assert_factory () 
  {
    bs_assert::asserter::set_factory (0);
  }

	virtual bs_assert::asserter *make (bool cond, const char *file, int line, const char *cond_str);

	void python_call_on_success (bool d)
	{
		disable_python_call_on_success = d;
	}

	py_assert_factory_setter *make_python (bool /*cond*/, const char * /*file*/, int /*line*/, const char * /*cond_str*/)
	{
		return new py_assert_factory_setter ();
	}
private:

	bool      disable_python_call_on_success;
};

void py_export_assert ();


} // namespace python
} // namespace blue_sky

#endif  // #ifndef PY_BS_ASSERT_H_

