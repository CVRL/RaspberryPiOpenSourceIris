from distutils.core import setup, Extension

module1 = Extension('bsif',
                    sources = ['bsif_wrapper.cpp','BSIFFilter.cpp'])

setup (name = 'bsif',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])