# Lisp in 99 lines of C and how to write one yourself

## Project code

To compile tinylisp:
~~~
$ cc -o tinylisp tinylisp-opt.c
~~~
The number of cells allocated is N=1024 by default, which is only 8K of memory.  To increase memory size, increase the value of N in the code.  Then recompile tinylisp.

To install one or more optional Lisp libraries to run tinylisp, use Linux/Unix `cat`:
~~~
cat common.lisp list.lisp math.lisp | ./tinylisp
~~~
But before you can do this, change the `look` function to reopen /dev/tty as explained in Section 7 of the [article](tinylisp.pdf).