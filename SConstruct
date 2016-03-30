# This is a main build script for BlueSky integration platform.

import os, os.path, glob;

# list of all sconscript files
# ORDER-SENSITIVE!
# add your sconscript only AFTER all dependent
ss_tree = [
		'kernel/SConscript',
		'python/bspy_loader/SConscript'
		];

# create custom variables telling whether to build debug and/or release
custom_vars = 0;
if os.path.exists('scons_vars.custom') :
	custom_vars = Variables(File('#scons_vars.custom').get_abspath());
else :
	custom_vars = Variables();

# what build kinds do we support?
def_build_kinds = ['debug', 'release'];
custom_vars.Add(ListVariable('build_kinds', 'List of supported build kinds', def_build_kinds[0], def_build_kinds));

# append some useful variables by default
#custom_vars.Add('nodeps', 'Set to 1 to build ignoring dependencies', '0');
custom_vars.Add(EnumVariable('deps', """Select how to track dependencies:
	auto - rely on SCons internal machinery,
	explicit - force using Depends(),
	off - build fully ignoring dependencies""",
	'auto', allowed_values = ['auto', 'explicit', 'off']));
custom_vars.Add('install', 'Set to 1 to install after build', '0');

# if user pointed to make install --- add path prefix variables describing where to install kernel and plugins
custom_vars.Add(PathVariable('prefix', 'Point where to install BlueSky kernel', Dir('#lib').get_abspath(),
	PathVariable.PathAccept));
custom_vars.Add(PathVariable('plugins_prefix', 'Point where to install BlueSky plugins', '$prefix/plugins',
	PathVariable.PathAccept));

custom_vars.Add('python_name', 'Put full Python interpreter name with version here, ex. python2.5', 'python2.5');
# add variable to decide whether to build with python support
custom_vars.Add('py', 'Set to 1 to build with Python support', '0');

custom_vars.Add(BoolVariable('auto_find_ss', 'Turn on automatic SConscripts search?', 0));

# search for platform-oriented scripts
platform_ss = glob.glob('scons_platform.*');
pnames = [''];
for p in platform_ss :
	ext = os.path.splitext(p)[1][1:];
	if len(ext) > 0 : pnames.append(ext);
custom_vars.Add(EnumVariable('platform', 'Specify the platform to build for', '', allowed_values = pnames));

custom_vars.Add(PathVariable('custom_script', 'Specify filename of custom build environment processing script',
	'scons_env.custom', PathVariable.PathAccept));

# create custom environment
custom_env = Environment(variables = custom_vars);
Export('custom_vars', 'custom_env');

def custom_proc_call() :
	# process custom settings
	if os.path.exists(custom_env['custom_script']) :
		SConscript(custom_env['custom_script']);

# extract build kinds specified by user
build_kinds = custom_env['build_kinds'];
Export('build_kinds');

# auto serach for SConscripts
def ss_search(root_dir, ss_prefix, ss_list) :
	nodes = os.listdir(root_dir);
	for node in nodes :
		abs_node = os.path.join(root_dir, node);
		if not os.path.isdir(abs_node) : continue;
#		print 'Travel path ', abs_node;
		ss_glob = glob.glob(os.path.join(abs_node, '[Ss][Cc]onscript'));
		if len(ss_glob) > 0 :
			ss = os.path.join(ss_prefix, node, os.path.basename(ss_glob[0]));
#			print 'Found ', ss;
			if ss not in ss_list : ss_list.append(ss);
		else :
			ss_search(abs_node, os.path.join(ss_prefix, node), ss_list);

if custom_env['auto_find_ss'] :
#	print "Auto-search on";
	root_dir = Dir('#').get_abspath();
#	print 'root_dir = ', root_dir;
	ss_list = [];
	ss_search(root_dir, '', ss_tree);
#	print ss_tree;
Export('ss_tree');

# setup commonly used names
plugin_dir = 'plugins';
build_dir = '#build';
exe_dir = '#exe';
Export('build_dir', 'exe_dir', 'plugin_dir');

# import some useful tools
SConscript('scons_tools');

# initialization stage is for correcting invariants, such as ss_list, etc
build_kind = 'init';
Export('build_kind');
# custom script call
custom_proc_call();
Import('*');

# dump ss_tree
print 'Processing build scripts:';
print '[';
for x in ss_tree :
	print x;
print']';

# configure
if not custom_env.GetOption('clean') and not custom_env.GetOption('help') :
	conf = Configure(custom_env);
	if not conf.CheckCXX() :
		print('!! Your compiler and/or environment is not correctly configured.');
		Exit(1);
	CheckLoki(conf);
	CheckBoost(conf);
	custom_env = conf.Finish();

# save default custom_env
custom_env_def = custom_env.Clone();

#debug
def compare_env(env1, env2) :
	print 'env1: ', env1;
	print 'env2: ', env2;
	d1 = env1.Dictionary(); d2 = env2.Dictionary();
	i = 0;
	for k in d1.keys() :
		if d1[k] != d2[k] :
			print 'env1[', k, '] = ', d1[k];
			print 'env2[', k, '] = ', d2[k];
			++i;
	if i == 0 : print 'Dictionary match!';
	print '';

# start global build cycle for every build kind
platform = custom_env['platform'];
for i in range(len(build_kinds)) :
	# inform everyone what are we building now
	build_kind = build_kinds[i];
	Export('build_kind');
	# format root build and exe paths
	tar_build_dir = os.path.join(build_dir, build_kind);
	tar_exe_dir = os.path.join(exe_dir, build_kind);
	# where plugin libs are expected to be after build?
	tar_exe_plugin_dir = os.path.join(tar_exe_dir, plugin_dir);
	Export('tar_build_dir', 'tar_exe_dir', 'tar_exe_plugin_dir');

	# reset custom_env to default values
	custom_env = custom_env_def.Clone();
	Export('custom_env');
	# invoke tuning scripts
	if len(platform) > 0 :
		SConscript('scons_platform.' + platform);
	custom_proc_call();
	Import('*');

	# add exe path to libraries search paths
	custom_env.AppendUnique(LIBPATH = [tar_exe_dir, tar_exe_plugin_dir]);
	if build_kind == 'debug' :
		custom_env.AppendUnique(CPPDEFINES = ['_DEBUG']);
	elif (build_kind == 'release') :
		custom_env.AppendUnique(CPPDEFINES = ['NDEBUG']);

	Export('custom_env');
	
	# parse scons files
	build_root = tar_build_dir;
	for j in range(len(ss_tree)) :
		# build in separate dir
		tar_build_dir = os.path.join(build_root, os.path.dirname(ss_tree[j]));
		inst_path = SConscript(ss_tree[j], variant_dir = tar_build_dir, duplicate = 0);
		# install to specified location
		#if not inst_path is None :
		#Install(tar_build_dir, os.path.join(tar_exe_dir, inst_path));
		#if scons_env['install'] == '1' :
	
	# Update template env with possibly cahnged build variables
	custom_vars.Update(custom_env_def);

# generate help text
Help(custom_vars.GenerateHelpText(custom_env));

