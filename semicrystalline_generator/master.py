#!/usr/bin/env python3
from os.path import abspath, dirname
import sys
import time
from subprocess import run, PIPE, Popen

def main():
    main_path = dirname(abspath(__file__)) + '/main.py '
    cmd = 'mpirun -n {} python3 -m mpi4py {}'.format(sys.argv[1], main_path) + ' '.join(sys.argv[2:])
    while True:
        process = Popen(cmd, stdout = PIPE, stderr = PIPE, shell = True)
        while process.poll() == None:
            output = process.stdout.readline()
            if output:
                sys.stdout.buffer.write(output)
        err_list = []
        err = process.stderr.readline()
        while err:
            print(err.decode())
            err_list.append(err.decode())
            err = process.stderr.readline()
        output = process.stdout.readline()
        # Print out the leftover of process output
        while output:
            sys.stdout.buffer.write(output)
            output = process.stdout.readline()
        err_msg = "".join(err_list)
        print('**********')
        print(err_msg)
        print('**********')
        if process.returncode == 0: 
            break
        elif ('abort' in err_msg or \
             'ABORT' in err_msg) and \
             process.returncode == 7:
            cmd = 'mpirun -n {} python3 -m mpi4py {}'.format(sys.argv[1], main_path) \
                    + ' '.join([arg for arg in sys.argv[2:] if arg != '--clean'])
            print("Atom missing error encountered, return code {}. Re-running the job.".format(process.returncode))
            print("Re-running command: {}".format(cmd))
        else:
            print("Some other error encountered, return code {}. Exiting...".format(process.returncode))
            sys.exit(process.returncode)



if __name__ == "__main__":
    main()
