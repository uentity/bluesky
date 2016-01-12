/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/


#ifndef __BS_ABSTRACT_STORAGE_H
#define __BS_ABSTRACT_STORAGE_H

#include "bs_common.h"
#include "bs_object_base.h"
//#include "smart_ptr.h"
//#include "storage_defines.h"
#include "bs_exception.h"
#include <iostream>

//-------------------- DEFINITIONS ----------------------------

#define STORAGE_IDENTIFIER(F) \
  bool identify(const std::string &str) const { \
  if( str == F ) return 1; \
  return 0; \
  }

// Avaliable value types
enum {
  ST_INT=1,
  ST_SHORT,
  ST_LONG,
  ST_UINT,
  ST_USHORT,
  ST_ULONG,

  ST_FLOAT,
  ST_DOUBLE,

  ST_CHAR,
  ST_UCHAR,

  ST_STRING
};

#define  ST_ACC_CREATE 1
#define  ST_ACC_RO 2
#define  ST_ACC_RW 4

#define TO_STRING(X) #X
#define STORED_VARIABLE(X)  \
  make_stored_variable(TO_STRING(X), &X)

#define ALLOW_ARCHIVE_ACCESS          \
  friend class hierarchial_oarchive;  \
  friend class hierarchial_iarchive;  \
  template< class ARCHIVE, typename OBJECT > friend void serialize(ARCHIVE &ar, OBJECT &obj);

// common serialization function
// you can assign specialization for specific object
template< typename ARCHIVE, typename OBJECT >
void serialize(ARCHIVE &ar, OBJECT &obj)
{
  obj.serialize(ar);
}

//-------------------------------------------------------------

namespace blue_sky
{
  class bs_abstract_storage;
  class empty_storage;
  class hierarchial_oarchive;
  template< typename T >  class stored_variable;

  class BS_API bs_abstract_storage : public objbase
  {
//    private:
//    protected:
    public:
      // Open storage
      virtual int open (const std::string& filename, int flags) = 0;

      // Close storage
      virtual int close () = 0;

      virtual bool is_opened () const = 0;

      // Identify storage format
      //virtual bool identify(const std::string& str) const = 0;

      // Creates a new group for object
      virtual int begin_object(const std::string& object_name) = 0;

      // Opens an existing group
      virtual int open_object(const std::string& object_name) = 0;

      // Closes current group
      virtual int end_object() = 0;

      // Write standalone values
      virtual int write(const std::string& name, int value) = 0;
      virtual int write(const std::string& name, float value) = 0;
      virtual int write(const std::string& name, double value) = 0;
      virtual int write(const std::string& name, char value) = 0;
      virtual int write(const std::string& name, const void *value, int type) = 0;

      // Write 1D array
      virtual int write(const std::string& name,
                          int type,
                          int size,
                          const void* data) = 0;

      // Write multidimensionsional array
      virtual int write(const std::string& name,
                          int type,
                          int rank,
                          const int* dimensions,
                          const void* data) = 0;

      virtual int read(const std::string& name, void *value, int type) = 0;
      // Read 1D array
      virtual int read(const std::string& name,
                          int type,
                          void *data) = 0;


      // Read multidimensionsional array
      /*virtual int read(const std::string& name,
                          int type,
                          int rank,
                          const int* dimensions,
                          void* data) = 0;*/

      // Get rank of data set. Returns number of dimension
      virtual int get_rank(const std::string& name) const = 0;

      // Get dimensions.
      virtual int get_dimensions(const std::string& name, int *dimensions) const = 0;
  };

  typedef bs_abstract_storage base_storage;
  typedef smart_ptr< bs_abstract_storage > sp_storage;

  class BS_API empty_storage : public bs_abstract_storage
  {
    BLUE_SKY_TYPE_DECL(empty_storage);

    int open (const std::string& /*filename*/, int /*flags*/) { return 0; }
    int close () { return 0; }
    bool is_opened () const { return 0; }
    bool identify(const std::string& /*str*/) const { return 0; }
    int begin_object(const std::string& /*object_name*/) { return 0; }
    int open_object(const std::string& /*object_name*/) { return 0; }
    int end_object() { return 0; }
    // write functions
    int write(const std::string& /*name*/, int /*value*/) { return 0; }
    int write(const std::string& /*name*/, float /*value*/) { return 0; }
    int write(const std::string& /*name*/, double /*value*/) { return 0; }
    int write(const std::string& /*name*/, char /*value*/) { return 0; }
    int write(const std::string& /*name*/, const void * /*value*/, int /*type*/) { return 0; }
    int write(const std::string& /*name*/, int /*type*/, int /*size*/, const void* /*data*/) { return 0; }
    int write(const std::string& /*name*/, int /*type*/, int /*rank*/, const int* /*dimensions*/, const void* /*data*/) { return 0; }
    int write(const std::string& /*name*/, void * /*value*/, int /*type*/) { return 0; }
    int read(const std::string& /*name*/, void * /*value*/, int /*type*/) { return 0; }
    int read(const std::string& /*name*/, int /*type*/, void * /*data*/) { return 0; }
    int get_rank(const std::string& /*name*/) const { return 0; }
    int get_dimensions(const std::string& /*name*/, int * /*dimensions*/) const { return 0; }
  };
  //BLUE_SKY_TYPE_STD_CREATE(empty_storage)


  //----------------------------------------------------------------------------
  // STORED VARIABLE
  //----------------------------------------------------------------------------
  template< typename T >
  class stored_variable
  {
  protected:
    //const char* name_;
    std::string name_;
    T* value_;
    bool free_;
  public:
    stored_variable(std::string &name, T* value, bool free_pointer = 0)
      : free_(free_pointer)
    {
      name_ = name;
      size_t dot_pos = name_.rfind('.');
      name = name_.substr(dot_pos+1, name_.size()-1);
      value_ = value;
    }
    stored_variable(std::string &name, T& value, bool free_pointer = 1)
      : free_(free_pointer)
    {
      name_ = name;
      size_t dot_pos = name_.rfind('.');
      name = name_.substr(dot_pos+1, name_.size()-1);
      value_ = new T(value);
    }
    stored_variable(stored_variable<T> &sv)
      : free_(sv.free_)
    {
      name_ = sv.name_;
      value_ = sv.value_;
    }
    ~stored_variable()
    {
      if(free_ == true && value_ != NULL)
        delete value_;
      name_ = "";
      value_ = NULL;
    }
    /*stored_variable(const stored_variable<T> *sv)
    {
      name_ = sv->name_;
      value_ = sv->value_;
    }*/

    stored_variable<T> &operator=(const stored_variable<T> &sv)
    {
      name_ = sv.name_;
      value_ = sv.value_;
      free_ = sv.free_;
      return *this;
    }
    //const char* name() const { return name_; }
    std::string &name() { return name_; }
    T* value() const { return value_; }
    void print() const
    {
			 std::cout << name_ << " = " << *value_ << std::endl;
    }
  };

  template< typename T >
  class __array
  {
  public:
    // constructor
    __array(size_t size)
      : size_(size), cur_pos_(0) { data_ = new T[size]; }
    __array(const __array<T> &ar)
    {
      size_ = ar.size_;
      data_ = new T[size_];
      memcpy(data_, ar.data_, sizeof(T)*size_);
      //data_
    }
    ~__array() {
      if( data_ != NULL )
        delete data_;
      data_ = NULL;
      size_ = 0;
    }
    size_t size() const { return size_; }

    // push/pop functions
    void push_back(T &item)
    {      data_[cur_pos_++] = item;    }
    void push_back(T item)
    {      data_[cur_pos_++] = item;    }
    void pop_back()
    {      cur_pos_--;    }

    // operators
    T &operator[](size_t index) const { return data_[index]; }
    __array<T> &operator=(const __array<T> &ar)
    {
      size_ = ar.size_;
      return *this;
    }
    // serialization
    template< class ARCHIVE >  void serialize(ARCHIVE &ar)
    {
      //ar & STORED_VARIABLE(size_);
      ar & this->make_stored_variable("size_", &size_);

    }
  protected:
    T *data_;
    size_t size_;
    size_t cur_pos_;
  };

  // Function that creates new stored variable and return it
  // make stored variable from pointer
  template< typename T >
  stored_variable<T> make_stored_variable(std::string name, T* value)
  {
    return stored_variable<T>(name, value);
  }
  // make stored variable from reference
  /*template< typename T >
  stored_variable<T> make_stored_variable(std::string name, T& value)
  {
    return stored_variable<T>(name, value);
  }*/

  /*template< typename T >
  stored_variable< __array<T> > make_stored_variable(std::string name, std::vector<T> *vect)
  {
    __array<T> my_array(vect->size());
    stored_variable< __array<T> > sv(name, my_array, true);
    return sv;
  }*/

  //----------------------------------------------------------------------------
  // ARCHIVES
  //----------------------------------------------------------------------------
  #define PUT_OPERATOR_OF_TYPE(TYPE, ST_TYPE)                    \
    void operator<<(stored_variable< TYPE >& var)                \
    { storage_.lock()->write(var.name(), var.value(), ST_TYPE); }

  #define GET_OPERATOR_OF_TYPE(TYPE, ST_TYPE)                    \
    void operator>>(stored_variable< TYPE >& var)                \
    { storage_.lock()->read(var.name(), (void*)var.value(), ST_TYPE); }

  class hierarchial_oarchive
  {
  public:
    // constructor
    hierarchial_oarchive() {}
    hierarchial_oarchive(sp_storage &storage)
      : storage_(storage) {}

    bool is_saving() { return true; }
    bool is_loading() { return false; }
    void set_storage(sp_storage &storage) { storage_ = storage; }

    template< typename T >
    void operator&(stored_variable< T >& var)
    {
      (*this) << var;
      //this->operator<<(var);
    }

    template< typename T >
    void operator<<(stored_variable< T >& var)
    {
      storage_.lock()->begin_object(var.name());
//#ifdef T::serialize
      //var.value()->serialize(*this);
//#else
      serialize(*this, *(var.value()));
//#endif
      storage_.lock()->end_object();
    }

    //void operator<<(stored_variable< float >& var) const
    //{ storage_.lock()->write(var.name(), var.value(), ST_FLOAT); }
    PUT_OPERATOR_OF_TYPE(int, ST_INT)
    PUT_OPERATOR_OF_TYPE(short int, ST_SHORT)
    PUT_OPERATOR_OF_TYPE(long int, ST_LONG)
    PUT_OPERATOR_OF_TYPE(unsigned int, ST_UINT)
    PUT_OPERATOR_OF_TYPE(unsigned short, ST_USHORT)
    PUT_OPERATOR_OF_TYPE(unsigned long, ST_ULONG)
    PUT_OPERATOR_OF_TYPE(char, ST_CHAR)
    PUT_OPERATOR_OF_TYPE(unsigned char, ST_UCHAR)
    PUT_OPERATOR_OF_TYPE(float, ST_FLOAT)
    PUT_OPERATOR_OF_TYPE(double, ST_DOUBLE)
    //PUT_OPERATOR_OF_TYPE(std::string, ST_STRING)

  private:
    sp_storage storage_;
  }; // hierarchial_oarchive

  //-----------------------------|

  class hierarchial_iarchive
  {
  public:
    // constructor
    hierarchial_iarchive() {}
    hierarchial_iarchive(sp_storage &storage)
      : storage_(storage) {}

    bool is_saving() { return false; }
    bool is_loading() { return true; }
    void set_storage(sp_storage &storage) { storage_ = storage; }

    template< typename T >
    void operator&(stored_variable< T >& var)
    {
      *this >> var;
    }

    template< typename T >
    void operator>>(stored_variable< T >& var)
    {
      storage_.lock()->begin_object(var.name());
      //var.value()->serialize(*this);
      serialize(*this, *(var.value()));
      storage_.lock()->end_object();
    }

    GET_OPERATOR_OF_TYPE(int, ST_INT)
    GET_OPERATOR_OF_TYPE(short int, ST_SHORT)
    GET_OPERATOR_OF_TYPE(long int, ST_LONG)
    GET_OPERATOR_OF_TYPE(unsigned int, ST_UINT)
    GET_OPERATOR_OF_TYPE(unsigned short, ST_USHORT)
    GET_OPERATOR_OF_TYPE(unsigned long, ST_ULONG)
    GET_OPERATOR_OF_TYPE(char, ST_CHAR)
    GET_OPERATOR_OF_TYPE(unsigned char, ST_UCHAR)
    GET_OPERATOR_OF_TYPE(float, ST_FLOAT)
    GET_OPERATOR_OF_TYPE(double, ST_DOUBLE)
    //PUT_OPERATOR_OF_TYPE(std::string, ST_STRING)

  private:
    sp_storage storage_;
  }; // hierarchial_oarchive



} // namespace blue-sky

#endif
