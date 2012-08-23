/*
 * Boctree implementation
 *
 * Copyright (C) Jan Elseberg
 *
 * Released under the GPL version 3.
 *
 */

#include "slam6d/Boctree.h"

//! Start-of-the-program initializer for the sequence map.
struct Initializer {
  Initializer() {
    for(unsigned char mask = 0; mask < 256; mask++) {
      for(unsigned char index = 0; index < 8; index++) {
        char c = 0;
        char *mimap = imap[index];  // maps area index to preference
        for(unsigned char i = 0; i < 8; i++) {
          if(( 1 << i ) & mask) {   // if ith node exists
            sequence2ci[index][mask][ mimap[i] ] = c++;
          } else {
            sequence2ci[index][mask][ mimap[i] ] = -1;
          }
        }
      }
      if (mask == UCHAR_MAX) break;
    }
  }
};

namespace{
  Initializer init;
}

char sequence2ci[8][256][8] = {};

char amap[8][8] = {
  {0, 1, 2, 4, 3, 5, 6, 7 },
  {1, 0, 3, 5, 2, 4, 6, 7 },
  {2, 0, 3, 6, 1, 4, 5, 7 },
  {3, 1, 2, 7, 0, 5, 4, 6 },
  {4, 5, 6, 0, 7, 1, 2, 3 },
  {5, 4, 7, 1, 6, 0, 3, 2 },
  {6, 4, 7, 2, 5, 0, 3, 1 },
  {7, 5, 6, 3, 4, 1, 2, 0 } };

char imap[8][8] = {
  {0, 1, 2, 4, 3, 5, 6, 7 },
  {1, 0, 4, 2, 5, 3, 6, 7 },
  {1, 4, 0, 2, 5, 6, 3, 7 },
  {4, 1, 2, 0, 6, 5, 7, 3 },
  {3, 5, 6, 7, 0, 1, 2, 4 },
  {5, 3, 7, 6, 1, 0, 4, 2 },
  {5, 7, 3, 6, 1, 4, 0, 2 },
  {7, 5, 6, 3, 4, 1, 2, 0 } };
