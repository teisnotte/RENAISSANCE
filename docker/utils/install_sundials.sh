#!/bin/bash

# Sundails 5.1.0
#wget https://github.com/LLNL/sundials/releases/download/v3.1.1/sundials-3.1.1.tar.gz
#tar -xzf sundials-3.1.1.tar.gz -C $HOME
#cd $HOME/sundials-3.1.1
#
#mkdir $HOME/sundials-3.1.1/builddir
#cd $HOME/sundials-3.1.1/builddir
#cmake -DLAPACK_ENABLE=ON \
#      -DSUNDIALS_INDEX_SIZE=64 \
#      -DCMAKE_INSTALL_PREFIX=$HOME/sundials-3.1.1 \
#      $HOME/sundials-3.1.1
#
#make
#make install
#
#
#ls $HOME/sundials-3.1.1/include

# Latest scikits odes distribution needs Sundails 5.1.0

wget https://github.com/LLNL/sundials/releases/download/v5.1.0/sundials-5.1.0.tar.gz
tar -xzf sundials-5.1.0.tar.gz -C $HOME

mv $HOME/sundials-5.1.0 $HOME/sundials
mkdir $HOME/build-sundials
cd $HOME/build-sundials/

cmake -DLAPACK_ENABLE=ON \
      -DSUNDIALS_INDEX_SIZE=64 \
      -DCMAKE_INSTALL_PREFIX=$HOME/sundials/ \
      $HOME/sundials/

make install

ls $HOME/sundials/include

