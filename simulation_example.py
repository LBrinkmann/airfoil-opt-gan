import os
import configparser
import pexpect
#import subprocess as sp
import gc
import numpy as np


def safe_remove(filename):
    if os.path.exists(filename):
        os.remove(filename)


def compute_coeff(_, reynolds=500000, mach=0, alpha=3, n_iter=200):

    try:
        # Has error: Floating point exception (core dumped)
        # This is the "empty input file: 'tmp/airfoil.log'" warning in other approaches
        child = pexpect.spawn('xfoil')
        timeout = 10

        ################
        #  turn off gui

        # child.expect('XFOIL   c> ', timeout)
        # child.sendline('')
        # child.expect('XFOIL   c> ', timeout)
        # child.sendline('PLOP')
        # child.expect('.* c> ', timeout)
        # child.sendline('G F')
        # child.expect('.* c> ', timeout)
        # child.sendline('')

        #  end turn off gui
        ################

        child.expect('XFOIL   c> ', timeout)
        child.sendline('load airfoil.dat')

        # print(str(child))
        child.expect('Enter airfoil name   s> ', timeout)
        child.sendline('af')
        # print(str(child))
        child.expect('XFOIL   c> ', timeout)
        child.sendline('OPER')
        # print(str(child))
        child.expect('.OPERi   c> ', timeout)
        child.sendline('VISC {}'.format(reynolds))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('ITER {}'.format(n_iter))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('MACH {}'.format(mach))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('PACC')
        child.expect(
            'Enter  polar save filename  OR  <return> for no file   s> ', timeout)
        child.sendline('airfoil.log')
        child.expect(
            'Enter  polar dump filename  OR  <return> for no file   s> ', timeout)
        child.sendline()
        child.expect('.OPERva   c> ', timeout)
        child.sendline('ALFA {}'.format(alpha))
        child.expect('.OPERva   c> ', timeout)
        child.sendline()
        child.expect('XFOIL   c> ', timeout)
        child.sendline('quit')

        child.expect(pexpect.EOF)
        child.close()

        # Has the dead lock issue
#        with open('tmp/control.in', 'w') as text_file:
#            text_file.write('load tmp/airfoil.dat\n' +
#                            'af\n' +
#                            'OPER\n' +
#                            'VISC {}\n'.format(reynolds) +
#                            'ITER {}\n'.format(n_iter) +
#                            'MACH {}\n'.format(mach) +
#                            'PACC\n' +
#                            'tmp/airfoil.log\n' +
#                            '\n' +
#                            'ALFA {}\n'.format(alpha) +
#                            '\n' +
#                            'quit\n')
#        os.system('xfoil <tmp/control.in> tmp/airfoil.out')

        # Has the dead lock issue
        # Has memory issue
#        ps = sp.Popen(['xfoil'], stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
#
#        # Use communicate() rather than .stdin.write, .stdout.read or .stderr.read
#        # to avoid deadlocks due to any of the other OS pipe buffers filling up and
#        # blocking the child process.
#        out, err = ps.communicate('load tmp/airfoil.dat\n' +
#                                  'af\n' +
#                                  'OPER\n' +
#                                  'VISC {}\n'.format(reynolds) +
#                                  'ITER {}\n'.format(n_iter) +
#                                  'MACH {}\n'.format(mach) +
#                                  'PACC\n' +
#                                  'tmp/airfoil.log\n' +
#                                  '\n' +
#                                  'ALFA {}\n'.format(alpha) +
#                                  '\n' +
#                                  'quit\n')

        res = np.loadtxt('airfoil.log', skiprows=12)

        if len(res) in [7, 9]:
            CL = res[1]
            CD = res[2]
        else:
            CL = -np.inf
            CD = np.inf

    except Exception as ex:
        #        print(ex)
        print('XFoil error!')
        print(child.before.decode())
        CL = -np.inf
        CD = np.inf
    else:
        print('XFoil success!')

    safe_remove(':00.bl')

    return CL, CD


def evaluate(return_CL_CD=False):

    # Airfoil operating conditions
    Config = configparser.ConfigParser()
    Config.read("op_conditions.ini")
    reynolds = float(Config.get('OperatingConditions', 'Reynolds'))
    mach = float(Config.get('OperatingConditions', 'Mach'))
    alpha = float(Config.get('OperatingConditions', 'Alpha'))
    n_iter = int(Config.get('OperatingConditions', 'N_iter'))

    CL, CD = compute_coeff(reynolds, mach, alpha, n_iter)
    perf = CL/CD

    if np.isnan(perf) or perf > 300:
        perf = np.nan
    if return_CL_CD:
        return perf, CL, CD
    else:
        return perf


if __name__ == "__main__":

    Config = configparser.ConfigParser()
    Config.read("op_conditions.ini")
    reynolds = float(Config.get('OperatingConditions', 'Reynolds'))
    mach = float(Config.get('OperatingConditions', 'Mach'))
    alpha = float(Config.get('OperatingConditions', 'Alpha'))
    n_iter = int(Config.get('OperatingConditions', 'N_iter'))

    CL, CD = compute_coeff(reynolds, mach, alpha, n_iter)
    print((CL, CD, CL/CD))
