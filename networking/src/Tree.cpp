#include "pch.h"
#include <networking/Tree.h>

#include <networking/Connection.h>

#include <Http.h>

#include <Request.h>
#include <Response.h>

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;
using namespace blue_sky::http_library;

using networking::Uri;

namespace
{
	struct MountPointInternal 
	{
		
	public:
		Path path;
		networking::Uri uri;
		thread_specific_ptr<ConnectionPtr> connection;

		MountPointInternal(Path p, networking::Uri u)
			: path(p), uri(u)
		{
		}

		MountPointInternal(const MountPointInternal & that)
			: path(that.path), uri(that.uri)
		{
		}
	};

	typedef std::list<MountPointInternal> MTab;
}

class Tree::TreeImpl
{
public:
	mutable boost::mutex mutex;

	ContextPtr context;
	MTab mtab;

	const MountPointInternal * get_mount_point(const Path & path) const
	{
		for (MTab::const_iterator it = mtab.begin(),
			end_it = mtab.end();
			it != end_it;
			++it)
		{
			if (path.equalto(it->path) || path.subfolderof(it->path))
			{
				return &*it;				
			}
		}
		return 0;
	}

	void get_mount_points_in(const Path & path, std::vector<string> & result) const
	{
		for (MTab::const_iterator it = mtab.begin(),
			end_it = mtab.end();
			it != end_it;
			++it)
		{
			if (it->path.subfolderof(path))
			{			
				Path p1 = it->path.subtract(path);
				result.push_back(*p1.begin());				
			}
		}
	}

	MountPointInternal * get_mount_point(const Path & path)
	{
		for (MTab::iterator it = mtab.begin(),
			end_it = mtab.end();
			it != end_it;
			++it)
		{
			if (path.equalto(it->path) || path.subfolderof(it->path))
			{
				return &*it;				
			}
		}
		return 0;
	}
	
	
	Uri path_to_uri(const Path & path)const
	{
		mutex::scoped_lock lock(mutex);

		const MountPointInternal * mp = get_mount_point(path);

		if (mp)
		{
			return mp->uri.append("objects").append(
					path.subtract(mp->path).str());				
		}
		else
		{
			throw bs_exception("Tree", "Path not found");
		}
	}

	Uri path_to_listing_uri(const Path & path)const
	{
		mutex::scoped_lock lock(mutex);

		const MountPointInternal * mp = get_mount_point(path);

		if (mp)
		{
			return mp->uri.append("listing").append(
					path.subtract(mp->path).str());				
		}
		else
		{
			throw bs_exception("Tree", "Path not found");
		}
	}

	void mount(const Path & path, const Uri & uri)
	{
		mutex::scoped_lock lock(mutex);

		for (MTab::const_iterator it = mtab.begin(),
			end_it = mtab.end();
			it != end_it;
			++it)
		{
			if (path.equalto(it->path))
				throw bs_exception("Tree", "Can't mount.");
		}
		MountPointInternal mp(path, uri);
		mtab.push_back(mp);
	}

	void umount(const Path & path)
	{
		mutex::scoped_lock lock(mutex);

		for (MTab::iterator it = mtab.begin(),
			end_it = mtab.end();
			it != end_it;
			++it)
			{
				if (it->path.equalto(path))
					mtab.erase(it++);
			}
	}

	ConnectionPtr get_connection(const Path & path)
	{
		mutex::scoped_lock lock(mutex);

		MountPointInternal * mp = get_mount_point(path);

		if (mp)
		{
			if (!(mp->connection.get()))
			{		
				mp->connection.reset(
					new ConnectionPtr(
						Connection::connect(
							context, mp->uri.str())));
			}	
			ConnectionPtr cnn = *(mp->connection.get());	
			return cnn;
		}
		else
		{
			throw bs_exception("Tree", "Path not found");
		}		
	}
};

Tree::Tree(ContextPtr context)
: pimpl(new TreeImpl())
{
	pimpl->context = context;
}

TreeNode Tree::get(const Path& path)
{
	string uri = pimpl->path_to_uri(path).str();
	
	return TreeNode(
			*this, 
			path,
			uri);
	
}

void Tree::mount(const networking::Uri& uri, const Path& path)
{
	pimpl->mount(path, uri);
}

void Tree::umount(const Path& path)
{
	pimpl->umount(path);
}

MountPointListPtr Tree::get_mount_points()
{
	MountPointListPtr result(new MountPointList());
	for (MTab::const_iterator it = pimpl->mtab.begin(),
		end_it = pimpl->mtab.end();
		it != end_it;
		++it)
	{
		result->push_back(MountPoint(it->path, it->uri));
	}
	return result;
}


void Tree::create(blue_sky::sp_obj obj, const blue_sky::networking::Path &path)
{
	Uri u = pimpl->path_to_uri(path);
	shared_ptr<Connection> cnn = Connection::connect(pimpl->context, u.str());
	Request req(pimpl->context);
	Response resp(pimpl->context);

	req.method = Request::M_PUT;
	req.Body = obj;
	req.uri = u;
	cnn->send(req, resp);

	if (resp.Status_code >= 400)
		throw bs_exception("Tree", resp.Reason_Phrase.c_str());
}

void Tree::erase(const Path&path)
{
	Uri u = pimpl->path_to_uri(path);
	shared_ptr<Connection> cnn = Connection::connect(pimpl->context, u.str());

	Request req(pimpl->context);
	Response rsp(pimpl->context);

	req.method = Request::M_DELETE;	
	req.uri = u;
	cnn->send(req, rsp);
	if (rsp.Status_code >= 400)
		throw bs_exception("Tree", rsp.Reason_Phrase.c_str());	
}

sp_folder_listing Tree::list(const Path&path)
{
	Uri u = pimpl->path_to_listing_uri(path);
	shared_ptr<Connection> cnn = Connection::connect(pimpl->context, u.str());

	Request req(pimpl->context);
	Response resp(pimpl->context);

	req.method = Request::M_GET;	
	req.uri = u;

	cnn->send(req, resp);
	if (resp.Status_code >= 400)
		throw bs_exception("Tree", resp.Reason_Phrase.c_str());

	sp_folder_listing result = resp.Body;
	
	std::vector<string> mps;
	pimpl->get_mount_points_in(path, mps);

	lsp_folder_listing locked(result);

	{
		for (std::vector<string>::const_iterator it = mps.begin(),
			end_it = mps.end();
			it != end_it;
			++it)
			{
				listing_entry_t entry = {*it, true};
				locked->add(entry);
			}
	}

	return result;
}

sp_obj Tree::get_object(const blue_sky::networking::Path &path)
{
	Uri u = pimpl->path_to_uri(path);
	
	ConnectionPtr cnn = pimpl->get_connection(path);
	Request req(pimpl->context);	
	req.method = Request::M_GET;	
	req.uri = u;

	Response resp(pimpl->context);
	
	cnn->send(req, resp);

	switch (resp.Status_code)
	{
	case http::STATUS_200_OK:
		return resp.Body;
		break;
	default:
		throw bs_exception("Tree", (string("Can't get object: ") + resp.Reason_Phrase).c_str());
	}
}

sp_obj Tree::lock_and_get(const blue_sky::networking::Path &path)
{
	//std::cerr << "Locking " << path.str() << std::endl;
	Uri u = pimpl->path_to_uri(path);

	ConnectionPtr cnn = pimpl->get_connection(path);
	Request req(pimpl->context);
	req.method = Request::M_LOCK;
	req.uri = u;

	Response resp(pimpl->context);

	cnn->send(req, resp);

	switch (resp.Status_code)
	{
	case http::STATUS_200_OK:
	case http::STATUS_204_No_Content:
		break;
	default:
		throw bs_exception("Tree", (string("Can't lock object: ") + resp.Reason_Phrase).c_str());
	}

	return get_object(path);
}

void Tree::put_and_unlock(const blue_sky::networking::Path &path, sp_obj obj)
{
	//std::cerr << "Unlocking " << path.str() << std::endl;
	Uri u = pimpl->path_to_uri(path);

	ConnectionPtr cnn = pimpl->get_connection(path);
	Request req(pimpl->context);
	req.method = Request::M_PUT;
	req.uri = u;
	req.Body = obj;

	Response resp(pimpl->context);

	cnn->send(req, resp);

	switch (resp.Status_code)
	{
	case http::STATUS_200_OK:
	case http::STATUS_204_No_Content:
		break;
	default:
		throw bs_exception("Tree", (string("Can't put object: ") + resp.Reason_Phrase).c_str());
	}

	req.method = Request::M_UNLOCK;
	req.uri = u;
	req.Body = 0;

	cnn->send(req, resp);

	switch (resp.Status_code)
	{
	case http::STATUS_200_OK:
	case http::STATUS_204_No_Content:
		break;
	default:
		throw bs_exception("Tree", (string("Can't unlock object: ") + resp.Reason_Phrase).c_str());
	}
}

void Tree::close_connections()
{
	mutex::scoped_lock lock(pimpl->mutex);

	for (MTab::iterator it = pimpl->mtab.begin(),
		end_it = pimpl->mtab.end();
		it != end_it;
		++it)
		{
			if (it->connection.get() != 0)
				it->connection->get()->close();
		}
}

obj_ro Tree::open_ro(const blue_sky::networking::Path &path)
{
	return obj_ro(*this, path);
}

obj_rw Tree::open_rw(const blue_sky::networking::Path & path)
{
	return obj_rw(*this, path);
}