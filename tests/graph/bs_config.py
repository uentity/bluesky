import os
import traceback

exported_envs = dict()

def exec_config(cfile):
    exports = dict ()

    for i in cfile:
        msg = ''
        for j in i:
            if j == '#' or j == '\n':
                break
            else:
                msg += j
        if msg != '' and msg != '\n':
            lr = ['','']
            idx = 0
            for j in i:
                if j == '=':
                    idx = 1
                elif j in ' \'"\n[]()':
                    continue
                elif j in ',;':
                    lr.append('')
                    idx += 1
                else:
                    lr[idx] += j

            exports[lr[0]] = lr[1:]

    for i in exports:
        export_str = ''
        for k in range(len(exports[i])):
            if exports[i][k] in os.environ.keys():
                exports[i][k] = os.environ[exports[i][k]]
            elif exports[i][k] in exports.keys():
                exports[i][k] = exports[exports[i][k]]


            if k != 0:
                if os.name == 'posix':
                    export_str += ':'
                else:
                    export_str += ';'

            if type(exports[i][k]) == type(list()):
                for ex in range(len(exports[i][k])):
                    if ex != 0:
                        if os.name == 'posix':
                            export_str += ':'
                        else:
                            export_str += ';'

                    export_str += exports[i][k][ex]
            else:
                export_str += exports[i][k]

        os.environ[i] = export_str
        exported_envs[i] = export_str

if os.name == 'posix':
	conf_fname = 'bs_config'
else:
	conf_fname = 'bs_config.win'

def configure(filename = conf_fname):
    try:
        conf_file = open(filename,'r')
    except:
        traceback.print_exc()
                        
    if conf_file > 0:
        exec_config(conf_file)
        conf_file.close()
    else:
        msg = 'No such file: "' + filename + '"'
        print msg

def create_export_bash_file():
    configure()
    ex_str = '#!/bin/bash\n\n'
    for i in exported_envs:
        ex_str += 'export ' + i + '=' + exported_envs[i] + '\n'

    try:
        ex_file = open('export_bs_envs.sh','w')

        if ex_file > 0:
            ex_file.write(ex_str)

        ex_file.close()
    except:
        traceback.print_exc()
