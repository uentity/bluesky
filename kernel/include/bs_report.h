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
//#include "bs_object_base.h"
#include "bs_messaging.h"
#include "bs_log_stream.h"

#include "loki/Singleton.h"

//#include <boost/thread/condition_variable.hpp>

namespace blue_sky {

	//typedef enum {
	//	OUT_LOG, //!< output to "$(BS_KERNEL_DIR)/blue_sky.log" file
	//	ERR_LOG, //!< output to "(BS_KERNEL_DIR)/errors.log" file
	//} jours;

#define OUT_LOG "out"
#define ERR_LOG "err"

	struct BS_API priority {
	  priority(int v = -1,int p = -1) : sect(v),prior(p) {}
		int sect,prior;
	};


	class BS_API bs_channel : public bs_refcounter {
		friend class bs_log;
	public:
		typedef smart_ptr< bs_channel, true > sp_channel;
		typedef smart_ptr< log::bs_stream >   sp_stream;
		typedef std::list< sp_stream >        sp_scr_list;
		typedef std::map<int, int>            msect;
		typedef msect::const_iterator         msect_const_iterator;
		typedef msect::iterator               msect_iterator;

		bool attach(const sp_stream&);
		bool detach(const sp_stream&);

		template <class T> sp_channel operator<<(const T &data_) {
			if (can_output) {
				*(buf_.lock()) << data_;
				if (!wait_end)
					send_to_subscribers();
			}
			return sp_channel(this);
		}

		bs_channel &add_section(int);
		bs_channel &rem_section(int);
		bs_channel &set_priority(priority);
		void set_can_output(bool);

		sp_channel operator<<(priority);

		sp_channel operator<<(sp_channel(*)(const sp_channel&));

		friend bool operator == (const bs_channel &lc, const bs_channel &rc);
		friend bool operator == (const bs_channel &lc, const std::string &rstr);
		friend bool operator < (const bs_channel &lc, const bs_channel &rc);
		friend bool operator < (const bs_channel &lc, const std::string &rstr);

		~bs_channel() {}

		void send_to_subscribers();

    //protected:
		//! ctors
		bs_channel();
		bs_channel(const std::string&);

		//! copy ctor
		bs_channel(const bs_channel&);
		bs_channel(const bs_channel &src, const std::string &tname);

		//! Assignment operator
		const bs_channel &operator=(const bs_channel&);

		void set_output_time();
		void set_wait_end();
		bool outputs_time() const;
		bool waits_end() const;

		void set_prefix(const std::string&);

		std::string get_name() const {return name;}

		void dispose() const;

	protected:
		std::map<int,int>               msects; //priority_pair rpair;
		bool                            output_time;
    bool                            wait_end;
    bool                            can_output;
		static int                      counter;
		std::string                     name;
		smart_ptr<std::ostringstream>   buf_;
		sp_scr_list                     scribes;
		std::string                     prefix;

    friend struct locked_channel;
	};
	typedef bs_channel::sp_channel sp_channel;
	typedef bs_channel::sp_stream sp_stream;

	template<class T>
	sp_channel operator<<(const sp_channel &ch, const T &what) {
		return *(ch.lock()) << what;
	}

	BS_API sp_channel operator<<(const sp_channel &ch, sp_channel(*what)(const sp_channel&));

  struct BS_API locked_channel
  {
    locked_channel (const sp_channel &ch)
    : ch_ (ch)
    , locked_buf_ (ch_->buf_)
    , buf_ (*locked_buf_)
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


    template <typename T>
    locked_channel &operator << (const T &what)
    {
        output (what);
        return *this;
    }

    locked_channel &operator<< (locked_channel &(*fun) (locked_channel &));

    locked_channel &operator<< (const priority &);

    sp_channel
    get_channel () const
    {
      return ch_;
    }

  private:
    lsmart_ptr <sp_channel>                       ch_;
    lsmart_ptr <smart_ptr <std::ostringstream> >  locked_buf_;
    std::ostringstream                            &buf_;
  };


	class BS_API bs_log : public bs_messaging {
		friend struct bs_private::log_wrapper;
		friend class thread_log;
		friend class std::map<int,bs_log>;
	public:
		typedef std::map <std::string, sp_channel>  channel_map_t;
		typedef channel_map_t::iterator             schan_iter;
		typedef channel_map_t::const_iterator       schan_iter_const;

		BLUE_SKY_SIGNALS_DECL_BEGIN(bs_messaging)
			log_channel_added,
			log_channel_removed,
		BLUE_SKY_SIGNALS_DECL_END

		sp_channel add_channel(const sp_channel&);
		bool rem_channel(const std::string&);

		locked_channel operator[] (const std::string &name_) const;
    locked_channel get_locked (const std::string &channel_name) const;

		void dispose() const;

		virtual ~bs_log() {}
		bs_log();

	private:
		channel_map_t channel_map_;
	};


	class BS_API thread_log {
		friend struct bs_private::thread_log_wrapper;
	public:
		typedef smart_ptr<bs_log>               sp_log;
		typedef std::map<int, sp_log>           mlog;
		typedef smart_ptr<mlog>                 sp_mlog;
		typedef mlog::const_iterator            const_iterator;
		typedef mlog::iterator                  iterator;

		/*BLUE_SKY_SIGNALS_DECL_BEGIN(thread_log)
			log_channel_added,
			log_channel_removed,
			log_stream_added,
			log_stream_removed,
		BLUE_SKY_SIGNALS_DECL_END*/

		sp_channel add_log_channel(const std::string&);
		sp_channel add_log_channel(const sp_channel&);
		bool add_log_stream(const std::string&,const sp_stream&);
		bool rem_log_channel(const std::string&);
		bool rem_log_stream(const std::string&,const sp_stream&);

		void kill();

		locked_channel operator[](const std::string&);

		thread_log();
		//thread_log(const thread_log&);

	private:
		sp_mlog logs;
	};

	//typedef singleton< bs_log > give_log;
	//typedef singleton< thread_log > give_tlog;

	BS_API sp_channel     bs_end(const sp_channel&);
  BS_API locked_channel &bs_end (locked_channel &ch);

	BS_API sp_channel     output_time(const sp_channel&);
  BS_API locked_channel &output_time (locked_channel &ch);

	BS_API sp_channel     wait_end(const sp_channel&);
  
	inline bool operator == (const bs_channel &lc, const bs_channel &rc) {
		return (lc.name == rc.name);
	}

	inline bool operator < (const bs_channel &lc, const bs_channel &rc) {
		return (lc.name < rc.name);
	}

	inline bool operator == (const bs_channel &lc, const std::string &rc_name) {
		return (lc.name == rc_name);
	}

	inline bool operator < (const bs_channel &lc, const std::string &rc_name) {
		return (lc.name < rc_name);
	}

}

#define BSOUT   BS_KERNEL.get_log ().get_locked (OUT_LOG)
#define BSERR   BS_KERNEL.get_log ().get_locked (ERR_LOG)

//! deprecated
#define BSERROR BS_KERNEL.get_log ().get_locked (ERR_LOG)

//#define BSOUT   blue_sky::give_log::Instance().get_locked (OUT_LOG)
//#define BSERROR blue_sky::give_log::Instance().get_locked (ERR_LOG)

#endif // _BS_REPORTER_H
