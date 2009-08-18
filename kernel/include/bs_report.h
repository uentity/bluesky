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

#include <list>
#include <set>
#include <map>
#include "bs_common.h"
#include "bs_refcounter.h"
#include "bs_messaging.h"
#include "bs_log_stream.h"

#include "loki/Singleton.h"

#include "throw_exception.h"

namespace blue_sky {

#define OUT_LOG "out"
#define ERR_LOG "err"

	struct BS_API priority {
	  priority (int section = -1, int priority = -1) 
      : sect (section)
      , prior (priority) 
    {
    }

		int sect;
    int prior;
	};


	class BS_API bs_channel : public bs_refcounter {
		friend class bs_log;
	public:
		typedef smart_ptr< bs_channel, true > sp_channel;

		typedef smart_ptr <log::bs_stream>    sp_stream_t;
		typedef std::vector <sp_stream_t>     stream_list_t;
    typedef stream_list_t::iterator       stream_iterator_t;

		typedef std::map <int, int>           sections_t;
		typedef sections_t::const_iterator    section_iterator_const_t;
		typedef sections_t::iterator          section_iterator_t;

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

		template <class T> sp_channel operator<<(const T &data) {
			if (can_output) 
        {
          buf_ << data;
          if (!wait_end)
            send_to_subscribers();
			  }

			return sp_channel(this);
		}

		sp_channel operator<< (const priority &p);
		sp_channel operator<< (functor_t functor);

		bs_channel &add_section (int section, int level);
		bs_channel &rem_section (int section);
		bs_channel &set_priority (const priority &p);

		void set_can_output (bool);
		void send_to_subscribers();
		void set_output_time();
		void set_wait_end();
		bool outputs_time() const;
		bool waits_end() const;

		void set_prefix(const std::string&);

		std::string get_name() const {return name;}
    bool get_can_output () const {return can_output;}

		void dispose() const;

	protected:
		sections_t                      sections_; 
		bool                            output_time;
    bool                            wait_end;
    bool                            can_output;
		std::string                     name;
		std::ostringstream              buf_;
		std::string                     prefix;

		stream_list_t                   output_streams_;

    friend struct locked_channel;
	};

	typedef bs_channel::sp_channel  sp_channel;
	typedef bs_channel::sp_stream_t sp_stream;
	typedef lsmart_ptr< sp_channel > lsp_channel;

	template<class T>
	sp_channel operator<<(const sp_channel &ch, const T &what) {
		return *(ch.lock()) << what;
	}

	BS_API sp_channel operator<<(const sp_channel &ch, sp_channel(*what)(const sp_channel&));

  struct proxy_log_end {};
  struct proxy_log_line {};
  struct proxy_log_output_time {};
  static void bs_end (const proxy_log_end &)
  {
  }

  static void bs_line (const proxy_log_line &)
  {
  }

  static void output_time (const proxy_log_output_time &)
  {
  }

  namespace detail {

    template <typename T>
    static std::string
    get_str (const T &t)
    {
      return boost::lexical_cast <std::string> (t);
    }

    //template <>
    //struct get_str <boost::format>
    //{
    //  static std::string
    //  get (const boost::format &t)
    //  {
    //    return t.str ();
    //  }
    //};

    static std::string
    get_str (int t)
    {
      return boost::lexical_cast <std::string> (t);
    }

    static std::string
    get_str (const double &t)
    {
      return boost::lexical_cast <std::string> (t);
    }

    static std::string
    get_str (const char *t)
    {
      return t;
    }

    static std::string
    get_str (const std::string &t)
    {
      return t;
    }
  }

  template <typename log_t, typename what_t>
  struct proxy_log
  {
    proxy_log (log_t &log, const what_t &what, bool can_output)
      : log_ (log)
      , what_ (what)
      , finished_ (false)
      , can_output_ (can_output)
    {
    }

    ~proxy_log ()
    {
      if (!finished_)
        {
          log_.log_output_not_finished ();
        }
    }

    bool get_can_output () const {return can_output_;}

    void
    output (const std::string &w, bool is_end = true)
    {
      if (log_.get_can_output ())
        log_.output (detail::get_str (what_) + w, is_end);
      else
        log_.output ("", is_end);
    }

    void
    log_output_not_finished ()
    {
      log_.log_output_not_finished ();
    }

    template <typename T>
    proxy_log <proxy_log <log_t, what_t>, T>
    operator << (const T &w)
    {
      finished_ = true;
      return proxy_log <proxy_log <log_t, what_t>, T> (*this, w, log_.get_can_output ());
    }

    void
    operator << (void (*) (const proxy_log_end &))
    {
      finished_ = true;
      if (log_.get_can_output ())
        log_.output (detail::get_str (what_));
      else
        log_.output ("");
    }

    void
    operator << (void (*) (const proxy_log_line &))
    {
      finished_ = true;
      if (log_.get_can_output ())
        log_.output (detail::get_str (what_), false);
      else
        log_.output ("", false);
    }

    log_t         &log_;
    const what_t  &what_;
    bool          finished_;
    bool          can_output_;
  };

  struct BS_API locked_channel
  {
    locked_channel (const sp_channel &ch, const char *file, int line)
    : ch_ (ch)
    , buf_ (ch_->buf_)
    , file_ (file)
    , line_ (line)
    {
    }

    void
    bs_end ();

    template <typename T>
    void
    output (const T &what)
    {
      if (ch_->can_output)
        {
          buf_ << what;
        }
    }

    bool
    get_can_output () const
    {
      return ch_->get_can_output ();
    }

    locked_channel &
    operator () (int section, int level);

    template <typename T>
    proxy_log <locked_channel, T>
    operator << (const T &what)
    {
      return proxy_log <locked_channel, T> (*this, what, ch_->get_can_output ());
    }

    locked_channel &
    operator << (void (*) (const proxy_log_output_time &))
    {
      ch_->set_output_time ();
      return *this;
    }

    void
    operator << (void (*) (const proxy_log_end &))
    {
      bs_end ();
    }

    lsp_channel
    get_channel () const
    {
      return ch_;
    }

    locked_channel &
    set_priority (int section, int level)
    {
      ch_->set_priority (priority (section, level));
      return *this;
    }

    void
    output (const std::string &w, bool is_end = true)
    {
      if (ch_->can_output)
        {
          buf_ << w;

          if (is_end)
            bs_end ();
        }
    }

    void
    log_output_not_finished ()
    {
      bool can_output = ch_->can_output;
      ch_->can_output = true;
      buf_ << "INVALID OUTPUT TO LOG: FILE (" << file_ << "), LINE (" << line_ << ")";
      bs_end ();

      ch_->can_output = can_output;
    }

  private:
    lsp_channel         ch_;
    std::ostringstream  &buf_;
    const char          *file_;
    int                 line_;
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
    get_locked (const std::string &channel_name, const char *file, int line) const;

		void 
    dispose() const;

		virtual ~bs_log() {}
		bs_log();

	private:
		channel_map_t channel_map_;
	};


	BS_API sp_channel
  wait_end(const sp_channel&);
  
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

