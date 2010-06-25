#!/bin/sh

if [ ! -f Makefile.options ]; then
  echo "   (+)  Creating Makefile.options"
  sed -e "s:PWD:`pwd`:" Makefile.options.in > Makefile.options
fi

