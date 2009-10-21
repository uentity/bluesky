/**
 * \file print_python_exception.h
 * \brief
 * \author Sergey Miryanov
 * \date 21.10.2009
 * */
#ifndef BLUE_SKY_PRINT_PYTHON_EXCEPTION_H_
#define BLUE_SKY_PRINT_PYTHON_EXCEPTION_H_

#include <string>
#include <boost/python.hpp>

namespace blue_sky {
namespace python {
namespace detail {

  struct py_object_handler 
  {
    py_object_handler (PyObject *obj)
    : obj (obj)
    {
    }

    void
    operator= (PyObject *p)
    {
      if (obj != Py_None)
        Py_XDECREF (obj);

      obj = p;
    }

    operator PyObject* ()
    {
      return obj;
    }

    ~py_object_handler ()
    {
      if (obj != Py_None)
        Py_XDECREF (obj);
    }

    PyObject *obj;
  };

  struct py_str : py_object_handler
  {
    py_str (PyObject *p)
    : py_object_handler (PyObject_Str (p))
    {
    }
  };

  struct py_attr : py_object_handler
  {
    py_attr (PyObject *p, const char *attr)
    : py_object_handler (PyObject_GetAttrString (p, attr))
    {
    }
  };

  void
  print_python_exception ()
  {
    PyObject *ptype = NULL, *pvalue = NULL, *ptraceback = NULL;
    PyErr_Fetch (&ptype, &pvalue, &ptraceback);

    using namespace boost::python;

    handle <> type_handler (ptype);
    handle <> value_handler (pvalue);
    handle <> traceback_handler (ptraceback);

    std::string type_str = extract <std::string> (py_str (ptype));
    std::string value_str = extract <std::string> (py_str (pvalue));
    BSERROR << "Exception message: " << type_str << " - " << value_str << bs_end;

    py_object_handler frame (PyObject_GetAttrString (ptraceback, "tb_frame"));
    if (frame != Py_None)
      {
        BSERROR << "Traceback: " << bs_end;
      }
    else
      {
        BSERROR << "No traceback!" << bs_end;
      }
    while (frame != Py_None)
      {
        py_attr line (frame, "f_lineno");
        py_attr code (frame, "f_code");
        py_attr file (code, "co_filename");
        py_attr name (code, "co_name");

        BSERROR << boost::format ("\t%s (filename: %s:%d)")
          % static_cast <std::string> (extract <std::string> (name))
          % static_cast <std::string> (extract <std::string> (file))
          % static_cast <int> (extract <int> (line))
          << bs_end;

        frame = PyObject_GetAttrString (frame, "f_back");
      }


    PyErr_Restore (ptype, pvalue, ptraceback);
  }

} // namespace detail 
} // namespace python
} // namespace blue_sky

#endif // #ifndef BLUE_SKY_PRINT_PYTHON_EXCEPTION_H_

