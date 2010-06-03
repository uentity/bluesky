// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#ifndef _BS_REPORTER_H
#define _BS_REPORTER_H

#include "bs_common.h"
#include "bs_messaging.h"
#include "bs_log_stream.h"
#include "bs_misc.h"
#include "throw_exception.h"

#include <list>
#include <set>
#include <map>

namespace blue_sky {

#define OUT_LOG "out"
#define ERR_LOG "err"

  class BS_API bs_channel : public bs_refcounter {
    friend class bs_log;
  public:
    typedef smart_ptr< bs_channel, true > sp_channel;

    typedef smart_ptr <log::bs_stream>    sp_stream_t;
    typedef std::vector <sp_stream_t>     stream_list_t;
    typedef stream_list_t::iterator       stream_iterator_t;

    typedef sp_channel (*functor_t) (const sp_channel &);

  public:
    //! ctors
    //bs_channel();
    bs_channel(const std::string&);

    //! copy ctor
    bs_channel(const bs_channel&);
    bs_channel(const bs_channel &src, const std::string &tname);

    ~bs_channel() {}

    //! Assignment operator
    const bs_channel &operator=(const bs_channel&);

    bool attach(const sp_stream_t &);
    bool detach(const sp_stream_t &);

    bs_channel &add_section (int section, int level);
    bs_channel &rem_section (int section);
    bs_channel &set_priority (const priority &p);

    void set_output_time();
    void set_prefix(const std::string&);

    std::string get_name() const {return name;}
    sp_stream_t get_stream (const std::string &name) const;

    void dispose() const;

  protected:
    bool                            output_time;
    bool                            wait_end;
    std::string                     name;
    //std::ostringstream              buf_;
    std::string                     prefix;

    stream_list_t                   output_streams_;

    friend struct locked_channel;
  };

  typedef bs_channel::sp_channel  sp_channel;
  typedef bs_channel::sp_stream_t sp_stream;
  typedef lsmart_ptr< sp_channel > lsp_channel;

  //template<class T>
  //sp_channel operator<<(const sp_channel &ch, const T &what) {
  //  return *(ch.lock()) << what;
  //}

  //BS_API sp_channel operator<<(const sp_channel &ch, sp_channel(*what)(const sp_channel&));

  struct proxy_log_end {};
  struct proxy_log_line {};
  struct proxy_log_output_time {};
  inline void bs_end (const proxy_log_end &)
  {
  }

  inline void bs_line (const proxy_log_line &)
  {
  }

  inline void output_time (const proxy_log_output_time &)
  {
  }

  template <typename log_t, typename proxy_t, typename what_t>
  struct proxy_log
  {
    proxy_log (log_t &log, proxy_t &proxy, const what_t &what)
      : log_ (log)
      , proxy_ (proxy)
      , what_ (what)
      , finished_ (false)
    {
    }

    ~proxy_log ()
    {
      if (!finished_)
        {
          log_.log_output_not_finished ();
        }
    }

    inline std::string
    what () const
    {
      return proxy_.what () + detail::get_str (what_);
    }

    template <typename T>
    inline proxy_log <log_t, proxy_log <log_t, proxy_t, what_t>, T>
    operator << (const T &w)
    {
      finished_ = true;
      return proxy_log <log_t, proxy_log <log_t, proxy_t, what_t>, T> (log_, *this, w);
    }

    inline void
    operator << (void (*) (const proxy_log_end &))
    {
      finished_ = true;
      log_.output (*this, true);
    }

    inline void
    operator << (void (*) (const proxy_log_line &))
    {
      finished_ = true;
      log_.output (*this, false);
    }

    log_t         &log_;
    proxy_t       &proxy_;
    const what_t  &what_;
    bool          finished_;
  };

  struct BS_API locked_channel
  {
    locked_channel (const sp_channel &ch, const char *file, int line)
    : ch_ (ch)
    , file_ (file)
    , line_ (line)
    , section_ (-1)
    , level_ (-1)
    {
    }

    inline std::string
    bs_end (std::string msg) const
    {
      if (ch_->output_time && msg.length())
        {
          std::string open_ = "[";
          std::string close_ = "]: ";

          msg = open_ + gettime () + close_ + ch_->prefix + msg + "\n";
        }
      else
        msg = msg + "\n";

      return msg;
    }

    inline locked_channel &
    operator () (int section, int level)
    {
      section_ = section;
      level_ = level;

      return *this;
    }

    template <typename T>
    inline proxy_log <locked_channel, locked_channel, T>
    operator << (const T &what)
    {
      return proxy_log <locked_channel, locked_channel, T> (*this, *this, what);
    }

    inline locked_channel &
    operator << (void (*) (const proxy_log_output_time &))
    {
      ch_->set_output_time ();
      return *this;
    }

    inline void
    operator << (void (*) (const proxy_log_end &))
    {
      output (*this);
    }

    inline lsp_channel
    get_channel () const
    {
      return ch_;
    }

    inline locked_channel &
    set_priority (int section, int level)
    {
      ch_->set_priority (priority (section, level));
      return *this;
    }

    inline const char *
    what () const
    {
      return "";
    }

    template <typename proxy_t>
    inline void
    output (const proxy_t &w, bool is_end = true)
    {
      for (size_t i = 0, cnt = ch_->output_streams_.size (); i < cnt; ++i)
        {
          const bs_channel::sp_stream_t &stream = ch_->output_streams_[i];
          if (stream->check_section (section_, level_))
            {
              if (is_end)
                stream->write (bs_end (w.what ()));
              else
                stream->write (w.what ());
            }
        }
    }

    inline void
    log_output_not_finished ()
    {
      for (size_t i = 0, cnt = ch_->output_streams_.size (); i < cnt; ++i)
        {
          const bs_channel::sp_stream_t &stream = ch_->output_streams_[i];
          stream->write (bs_end ("INVALID OUTPUT TO LOG: FILE (" + std::string (file_) + "), LINE (" + detail::get_str (line_) + ")"));
        }
    }

  private:
    lsp_channel         ch_;
    const char          *file_;
    int                 line_;
    int                 section_;
    int                 level_;
  };


  class BS_API bs_log : public bs_messaging 
  {
  public:
    typedef std::map <std::string, sp_channel>  channel_map_t;
    typedef channel_map_t::iterator             channel_iterator_t;
    typedef channel_map_t::const_iterator       channel_iterator_const_t;

    BLUE_SKY_SIGNALS_DECL_BEGIN(bs_messaging)
      log_channel_added,
      log_channel_removed,
    BLUE_SKY_SIGNALS_DECL_END

  public:
    sp_channel 
    add_channel(const sp_channel&);

    bool 
    rem_channel(const std::string&);

    locked_channel 
    get_locked (const std::string &name_, const char *file, int line) const
    {
      channel_iterator_const_t it = channel_map_.find(name_);
      if (it != channel_map_.end()) 
        {
          return locked_channel (it->second, file, line);
        }

      bs_throw_exception (std::string("Unknown log name ") + name_);
    }

    void 
    dispose() const;

    virtual ~bs_log() {}
    bs_log();

  private:
    channel_map_t channel_map_;
  };


  //BS_API sp_channel
  //wait_end(const sp_channel&);
  
  inline bool operator == (const bs_channel &lc, const bs_channel &rc) {
    return (lc.get_name () == rc.get_name ());
  }

  inline bool operator < (const bs_channel &lc, const bs_channel &rc) {
    return (lc.get_name () < rc.get_name ());
  }

  inline bool operator == (const bs_channel &lc, const std::string &rc_name) {
    return (lc.get_name () == rc_name);
  }

  inline bool operator < (const bs_channel &lc, const std::string &rc_name) {
    return (lc.get_name () < rc_name);
  }

} // namespace blue_sky

#include "bs_report_thread.h"

#define BSOUT   kernel::get_log ().get_locked (OUT_LOG, __FILE__, __LINE__)
#define BSERR   kernel::get_log ().get_locked (ERR_LOG, __FILE__, __LINE__)
#define BSERROR BSERR

#endif // _BS_REPORTER_H

