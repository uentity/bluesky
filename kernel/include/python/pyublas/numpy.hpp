//
// Copyright (c) 2008 Andreas Kloeckner
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//

// This file is part of BlueSky
// We need only numpy_array, so all other unrelated stuff was erased


#ifndef _AFAYFYDASDFAH_PYUBLAS_HEADER_SEEN_NUMPY_HPP
#define _AFAYFYDASDFAH_PYUBLAS_HEADER_SEEN_NUMPY_HPP




#include <cstdlib>
#include <numeric>
#include <pyublas/python_helpers.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <numpy/arrayobject.h>


namespace
{
  static struct _pyublas_array_importer
  {
    _pyublas_array_importer()
    { 
      import_array();
    }
  } _array_importer;
}


namespace pyublas
{
  inline NPY_TYPES get_typenum(bool) { return NPY_BOOL; }
  inline NPY_TYPES get_typenum(npy_bool) { return NPY_BOOL; }
  inline NPY_TYPES get_typenum(npy_byte) { return NPY_BYTE; }
  // NPY_TYPES get_typenum(npy_ubyte) { return NPY_UBYTE; }
  inline NPY_TYPES get_typenum(npy_short) { return NPY_SHORT; }
  inline NPY_TYPES get_typenum(npy_ushort) { return NPY_USHORT; }
  inline NPY_TYPES get_typenum(npy_int) { return NPY_INT; }
  inline NPY_TYPES get_typenum(npy_uint) { return NPY_UINT; }
  inline NPY_TYPES get_typenum(npy_long) { return NPY_LONG; }
  inline NPY_TYPES get_typenum(npy_ulong) { return NPY_ULONG; }
  inline NPY_TYPES get_typenum(npy_longlong) { return NPY_LONGLONG; }
  inline NPY_TYPES get_typenum(npy_ulonglong) { return NPY_ULONGLONG; }
  inline NPY_TYPES get_typenum(npy_float) { return NPY_FLOAT; }
  inline NPY_TYPES get_typenum(npy_double) { return NPY_DOUBLE; }
  inline NPY_TYPES get_typenum(npy_longdouble) { return NPY_LONGDOUBLE; }
  inline NPY_TYPES get_typenum(npy_cfloat) { return NPY_CFLOAT; }
  inline NPY_TYPES get_typenum(npy_cdouble) { return NPY_CDOUBLE; }
  inline NPY_TYPES get_typenum(npy_clongdouble) { return NPY_CLONGDOUBLE; }
  inline NPY_TYPES get_typenum(std::complex<float>) { return NPY_CFLOAT; }
  inline NPY_TYPES get_typenum(std::complex<double>) { return NPY_CDOUBLE; }
  inline NPY_TYPES get_typenum(std::complex<long double>) { return NPY_CLONGDOUBLE; }
  inline NPY_TYPES get_typenum(boost::python::object) { return NPY_OBJECT; }
  inline NPY_TYPES get_typenum(boost::python::handle<>) { return NPY_OBJECT; }
  /* NPY_STRING, NPY_UNICODE unsupported for now */

  template <class T>
  inline
  bool is_storage_compatible(PyObject *ary)
  {
    /* This piece of code works around the fact that 'int' and 
     * 'long int' are the same on 32-bit machines, which can lead
     * to typenum mismatches. Therefore, for integers, we only
     * compare size and signedness.
     */

    if (boost::is_integral<T>::value && PyArray_ISINTEGER(ary))
    {
      return (sizeof(T) == PyArray_ITEMSIZE(ary)
          && bool(boost::is_signed<T>::value) 
          == bool(PyArray_ISSIGNED(ary)));
    }
    else
      return PyArray_TYPE(ary) == get_typenum(T());
  }




  // tool functions -----------------------------------------------------------
  inline
  npy_intp size_from_dims(int ndim, const npy_intp *dims)
  {
    if (ndim != 0)
      return std::accumulate(dims, dims+ndim, 1, std::multiplies<npy_intp>());
    else
      return 1;
  }




  // ublas storage array ------------------------------------------------------
  template <class T>
  class numpy_array
  {
    private:
      // Life support for the numpy array.
      boost::python::handle<>         m_numpy_array;

    public:
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;
      typedef T value_type;
      typedef const T &const_reference;
      typedef T &reference;
      typedef const T *const_pointer;
      typedef T *pointer;

      // Construction and destruction
      numpy_array()
      { }

      numpy_array(size_type n)
      {
        npy_intp dims[] = { n };
        m_numpy_array = boost::python::handle<>(
            PyArray_SimpleNew(1, dims, get_typenum(T())));
      }

      numpy_array(int ndim_, const npy_intp *dims_)
      {
        m_numpy_array = boost::python::handle<>(
            PyArray_SimpleNew(
              ndim_, 
              const_cast<npy_intp *>(dims_), 
              get_typenum(T())));
      }

      numpy_array(size_type n, const value_type &v)
      {
        if (n)
        {
          npy_intp dims[] = { n };
          m_numpy_array = boost::python::handle<>(
              PyArray_SimpleNew(1, dims, get_typenum(T())));
          std::fill(begin(), end(), v);
        }
      }

      /* MISSING
      Range constructor 	X(i, j) 	
      i and j are Input Iterators whose value type is convertible to T 	X
      */

      numpy_array(const boost::python::handle<> &obj)
        : m_numpy_array(obj)
      {
        if (!obj.get())
          return;
        if (obj.get() == Py_None)
        {
          m_numpy_array = boost::python::handle<>();
          return;
        }

        if (!PyArray_Check(obj.get()))
          PYUBLAS_PYERROR(TypeError, "argument is not a numpy array");
        if (!is_storage_compatible<T>(obj.get()))
          PYUBLAS_PYERROR(TypeError, "argument is numpy array of wrong type");
        if (!PyArray_CHKFLAGS(obj.get(), NPY_ALIGNED))
            PYUBLAS_PYERROR(ValueError, "argument array is not aligned");
        if (PyArray_CHKFLAGS(obj.get(), NPY_NOTSWAPPED))
            PYUBLAS_PYERROR(ValueError, "argument array does not have native endianness");
        if (PyArray_ITEMSIZE(obj.get()) != sizeof(T))
            PYUBLAS_PYERROR(ValueError, "itemsize does not match size of target type");
      }

    private:
      void resize_internal (size_type new_size, value_type init, bool preserve = true) 
      {
        size_type old_size;
        if (m_numpy_array.get())
          old_size = size();
        else
        {
          preserve = false;
          old_size = 0;
        }

        if (new_size != old_size) 
        {
          npy_intp dims[] = { new_size };
          boost::python::handle<> new_array = boost::python::handle<>(
              PyArray_SimpleNew(1, dims, get_typenum(T())));
          pointer new_data = reinterpret_cast<T *>(
              PyArray_DATA(new_array.get()));

          if (preserve) 
          {
            std::copy(data(), data() + std::min(new_size, old_size), new_data);
            std::fill(new_data + std::min(new_size, old_size), new_data + new_size, init);
          }

          m_numpy_array = new_array;
        }
      }

    public:
      void resize (size_type size) 
      {
        resize_internal (size, value_type(), false);
      }
      void resize (size_type size, value_type init) 
      {
        resize_internal (size, init, true);
      }

      size_type size() const 
      { 
        if (!is_valid())
          return 0;

        if (ndim() != 0)
        {
          return end()-begin();
        }
        else
          return 1;
      }

      // metadata
      bool is_valid() const
      { return m_numpy_array.get(); }
      size_type ndim() const 
      { return PyArray_NDIM(m_numpy_array.get()); }
      const npy_intp *dims() const 
      { return PyArray_DIMS(m_numpy_array.get()); }
      npy_intp dim(npy_intp i) const 
      { return PyArray_DIM(m_numpy_array.get(), i); }
      const npy_intp *strides() const 
      { return PyArray_STRIDES(m_numpy_array.get()); }
      npy_intp stride(npy_intp i) const 
      { return PyArray_STRIDE(m_numpy_array.get(), i); }

      npy_intp itemsize() const
      { return sizeof(T); }
      bool writable() const
      { return PyArray_ISWRITEABLE(m_numpy_array.get()); }

      // shape manipulation
      void reshape(int ndim_, const npy_intp *dims_, 
          NPY_ORDER order=NPY_CORDER)
      {
        PyArray_Dims d = { const_cast<npy_intp *>(dims_), ndim_ };
        m_numpy_array = boost::python::handle<>(
            PyArray_Newshape(
              (PyArrayObject *) m_numpy_array.get(), &d, order));
      }

      // Raw data access
      T *data()
      {
        return reinterpret_cast<T *>(
            PyArray_DATA(m_numpy_array.get()));
      }

      const T *data() const
      {
        return reinterpret_cast<const T *>(
            PyArray_DATA(m_numpy_array.get()));
      }

      // Element access
      const_reference operator [] (size_type i) const 
      {
        BOOST_UBLAS_CHECK(i < size(), boost::numeric::ublas::bad_index());
        return begin()[i];
      }

      reference operator [] (size_type i) 
      {
        BOOST_UBLAS_CHECK(i < size(), boost::numeric::ublas::bad_index());
        return begin()[i];
      }

      // Assignment
      numpy_array &operator=(const numpy_array &a) 
      {
        m_numpy_array = a.m_numpy_array;
        return *this;
      }

      numpy_array &assign_temporary(numpy_array &a) 
      {
        m_numpy_array = a.m_numpy_array;
        return *this;
      }

        // Swapping
      void swap (numpy_array &a) 
      {
        if (this != &a)
          std::swap(m_numpy_array, a.m_numpy_array);
      }

      friend void swap(numpy_array &a1, numpy_array &a2) 
      {
        a1.swap (a2);
      }

      // Iterators simply are pointers.

      typedef const_pointer const_iterator;

    protected:

      npy_intp max_pos_stride_index() const
      {
        npy_intp current_idx = -1;
        npy_intp current_max = 0;
        for (unsigned i = 0; i < ndim(); ++i)
        {
          npy_intp si = stride(i);
          if (si > current_max)
          {
            current_max = si;
            current_idx = i;
          }
        }

        return current_idx;
      }

    public:
      const_iterator begin() const 
      {
        const_iterator result = data();
        for (unsigned i = 0; i < ndim(); ++i)
        {
          const npy_intp si = stride(i)/npy_intp(sizeof(T));
          const npy_intp di = dim(i);
          if (si < 0 && di)
            result += si*(di-1);
        }

        return result;
      }

      const_iterator end() const 
      { 
        const npy_intp mpsi = max_pos_stride_index();

        if (mpsi != -1)
        {
          const npy_intp mps = stride(mpsi)/npy_intp(sizeof(T));
          return data() + mps*dim(mpsi);
        }
        else
          return data()+1;
      }

      typedef pointer iterator;

      iterator begin() 
      {
        return const_cast<iterator>(
            const_cast<numpy_array const *>(this)->begin());
      }

      iterator end() 
      { 
        return const_cast<iterator>(
            const_cast<numpy_array const *>(this)->end());
      }

      // Reverse iterators
      typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
      typedef std::reverse_iterator<iterator> reverse_iterator;

      const_reverse_iterator rbegin() const 
      { return const_reverse_iterator(end()); }

      const_reverse_iterator rend() const 
      { return const_reverse_iterator(begin ()); }

      reverse_iterator rbegin() 
      { return reverse_iterator(end()); }

      reverse_iterator rend () 
      { return reverse_iterator(begin()); }

      // Data accessor

      const boost::python::handle<> handle() const
      {
        if (is_valid())
          return m_numpy_array;
        else
          return boost::python::handle<>(
              boost::python::borrowed(Py_None));
      }
  };

}

#endif

