this is the source code for proejct of CSED490C-01 on postech.

requires gmp-6.3.0 library in this directory
download and build it with following commands
```
wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
unxz gmp-6.3.0.tar.xz
tar -xvf gmp-6.3.0.tar
cd gmp-6.3.0/
mkdir build
./configure --prefix=`pwd`/build/
make install
```

then you can run the example by
```
./run_for_opt_levels.sh $n $k
```
where $n is the initial value, and $k is the number of threads.

you can change optimization level, # of numbers per thread via compilation macro OPT and MULT.
change it as you like in the shell script.

for bulk execution, you need to modify the source code (use `_main` instead of `main`)
