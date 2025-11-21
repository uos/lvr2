# Packaging Notes

At first we can simply create pre-compiled binaries of lvr2 as follows:

```bash
make package
```

results in a `lvr2-X-x86_64-linux.tar.gz` (X: version) which can be extracted afterwards to any root dir:

```bash
mkdir ~/software
tar -C ~/software -xzf lvr2-X-x86_64-linux.tar.gz
```

you can include this by adding updating the paths. For example in the `.bashrc` file:

```bash
export LVR2_INSTALL_DIR=~/software/lvr2-X-x86_64-linux
export LD_LIBRARY_PATH=$LVR2_INSTALL_DIR:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LVR2_INSTALL_DIR:$LIBRARY_PATH
export CMAKE_PREFIX_PATH=$LVR2_INSTALL_DIR:$CMAKE_PREFIX_PATH
```


