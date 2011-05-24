/**
 * \file py_bs_assert.cpp
 * \brief bs_assert python wrappers impl
 * \author Sergey Miryanov
 * \date 14.05.2008
 * */
#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

#include "py_bs_assert.h"
#include <cassert>

namespace blue_sky {
namespace python {

/** 
 * \brief class helper to call assert's function in python from c++
 * \detail
 *        because python is dynamic typied language we can work with
 *        python::object as with bs_assert::asserter 
 * */
class py_asserter_caller : public bs_assert::asserter
{
public:

	/** 
	 * \brief initialize caller. we have to increment python::object
	 *        reference to prevent python gc of collection of this object
	 * */
	py_asserter_caller (const boost::python::object &obj)
    : bs_assert::asserter (true, "", -1, "")
		, obj (obj)
	{
		Py_INCREF (obj.ptr ());
	}

	py_asserter_caller &operator = (const py_asserter_caller &p)
	{
		if (this != &p)
		{
			Py_DECREF (obj.ptr ());

			obj = p.obj;
			Py_INCREF (obj.ptr ());
		}

		return *this;
	}

	virtual ~py_asserter_caller ()
	{
		Py_DECREF (obj.ptr ());
	}

	/** 
	 * \brief set variable list and call handle that specified in python
	 **/
	virtual bool handle () const
	{
		boost::python::call_method <void> (obj.ptr (), "set_var_list", var_list);
		return boost::python::call_method <bool> (obj.ptr (), "handle");
	}

	/** 
	 * \brief return assert factory that was be set early 
	 *        (from python or c++)
	 * */
	static bs_assert::assert_factory *get_factory()
	{
		return bs_assert::asserter::factory ();
	}

private:

	boost::python::object obj;
};

/** 
 * \brief proxy to set factory in bs_assert::asserter
 * */
void py_assert_factory_setter::set_factory (py_assert_factory *factory)
{
	bs_assert::asserter::set_factory (factory);
}

/** 
 * \brief make an asserter that will be called from c++ (or python, see call_asserter)
 * \detail
 *        will be created through factory that defined in python
 * */
bs_assert::asserter *
py_assert_factory::make (bool b, const char *file, int line, const char *cond_str)
{
	if (disable_python_call_on_success && b)
		return new bs_assert::asserter (b, file, line, cond_str);

	boost::python::object o = this->get_override("make")(b, file, line, cond_str);
	if (o)
		return new py_asserter_caller (o);

	return new bs_assert::asserter (b, file, line ,cond_str);
}

/** 
 * \brief helper class to impl assert in python
 * */
struct call_assert
{
	call_assert(bool cond, const char *file, int line, const char *cond_str)
	{
		bs_assert::asserter *a = py_asserter_caller::get_factory()->make(cond, file, line, cond_str);
		if (a && !a->handle())
		{
			a->break_here ();
		}
	}
};

void py_export_assert ()
{
	using namespace boost::python;

	class_ <py_assert_factory_setter> ("bs_assert")
		.def ("set_factory", &py_assert_factory_setter::set_factory, with_custodian_and_ward<2, 1>())
		;

	class_ <py_assert_factory, boost::noncopyable> ("bs_assert_factory")
		.def ("make", boost::python::pure_virtual(&py_assert_factory::make_python), return_value_policy<manage_new_object> ())
		.def ("python_call_on_success", &py_assert_factory::python_call_on_success)
		;

	//class_ <call_assert, boost::noncopyable> ("bs_assert", init <bool,const char*,int, const char*>());
}

} // namespace python
} // namespace blue_sky

