/**
 * \file bs_log_stream.h
 * \brief 
 * \author Sergey Miryanov
 * \date 07.07.2009
 * */
#ifndef BS_LOG_STREAM_H_
#define BS_LOG_STREAM_H_

#include "bs_common.h"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

namespace blue_sky {

  struct BS_API priority {
    priority (int section = -1, int priority = -1) 
      : sect (section)
      , prior (priority) 
    {
    }

    int sect;
    int prior;
  };

namespace log {

    //there is error with smart_pointer access... I don't know why.
  class BS_API bs_stream { //: public bs_refcounter { //!!!!!!!!!!!!!!!!!!!!!!!
  public:

    typedef std::map <int, int>           sections_t;
    typedef sections_t::const_iterator    section_iterator_const_t;
    typedef sections_t::iterator          section_iterator_t;

  public:

    bs_stream (const std::string &name)
    : name_ (name)
    {
    }

    inline bool 
    check_section (int section, int level) const
    {
      if (sections_.empty ())
        {
          return true;
        }

      section_iterator_const_t iter = sections_.find (section);
      if (iter != sections_.end ())
        {
          if (iter->second < level || iter->second == -1)
            {
              return false;
            }
          else
            {
              return true;
            }
        }
      else
        {
          return false;
        }
    }

    const std::string &
    get_name () const
    {
      return name_;
    }

    void add_section (int section, int level);
    void rem_section (int section);

    void set_priority (const priority &p);

    virtual bool subscribe(bs_channel&);
    virtual bool unsubscribe(bs_channel&);
    virtual void write(const std::string&) const = 0; //{}//= 0;//{std::cout << "asdddd" << std::endl;}
    //virtual void dispose() const;

    virtual ~bs_stream() {}

  private:

    sections_t                      sections_; 
    std::string                     name_;

  };

} // namespace log
} // namespace blue_sky


#endif // #ifndef BS_LOG_STREAM_H_

