#include "pch.h"

#include <networking/Tree.h>
#include <networking/TreeNode.h>

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;

class obj_ro::Impl
{
public:
	Impl(Tree tree)
		: fTree(tree)
	{
	}
	
	Tree fTree;
	Path fPath;
	std::string fEtag;
	time_t fMtime;
};

obj_ro::obj_ro(blue_sky::networking::Tree tree, const blue_sky::networking::Path &path)
: pimpl(new Impl(tree))
{
	pimpl->fPath = path;
}

const sp_obj obj_ro::get()
{
	return pimpl->fTree.get_object(pimpl->fPath);
}

void obj_ro::reload()
{

}

obj_ro::~obj_ro()
{

}


class obj_rw::Impl
{
public:
	Impl(Tree tree)
		: fTree(tree)
	{

	}

	Tree fTree;
	Path fPath;
	sp_obj fObj;
	bool fIsClosed;

	void close()
	{
		if (!fIsClosed)
		{
			fTree.put_and_unlock(fPath, fObj);
			fIsClosed = true;
		}
	}

	~Impl()
	{
		close();
	}
};

obj_rw::obj_rw(blue_sky::networking::Tree tree, const blue_sky::networking::Path &path)
: pimpl(new Impl(tree))
{
	pimpl->fPath = path;
	pimpl->fObj = tree.lock_and_get(path);
	pimpl->fIsClosed = false;
}

sp_obj obj_rw::get()
{
	return pimpl->fObj;
}

void obj_rw::reload()
{
	pimpl->fObj = pimpl->fTree.get_object(pimpl->fPath);
}

void obj_rw::close()
{
	pimpl->close();
}

obj_rw::~obj_rw()
{
}



// TreeNode ================

class TreeNode::TreeNodeImpl
{
public:
	ContextPtr context;
	Path path;
	Uri uri;
	Tree tree;

	sp_obj obj;

	int lock_counter;

	TreeNodeImpl(Tree tree)
		:tree(tree), lock_counter(0)
	{

	}
};

TreeNode::TreeNode(
	Tree tree,
	const Path & path, 
	const Uri &uri)
	: pimpl(new TreeNode::TreeNodeImpl(tree))
{
	pimpl->path = path;
	pimpl->uri = uri;
	pimpl->tree = tree;	
}

TreeNode TreeNode::get(const blue_sky::networking::Path &path)const
{
	return pimpl->tree.get(pimpl->path.down(path));
}

Uri TreeNode::uri()
{
	return pimpl->uri;
}

void TreeNode::create(blue_sky::sp_obj obj)
{
	pimpl->tree.create(obj, pimpl->path);
}

void TreeNode::erase()
{
	pimpl->tree.erase(pimpl->path);
}

obj_ro TreeNode::open_ro()
{
	return pimpl->tree.open_ro(pimpl->path);
}

obj_rw TreeNode::open_rw()
{
	return pimpl->tree.open_rw(pimpl->path);
}

sp_folder_listing TreeNode::list()
{
	return pimpl->tree.list(pimpl->path);
}










