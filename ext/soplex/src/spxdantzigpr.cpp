/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the class library                   */
/*       SoPlex --- the Sequential object-oriented simPlex.                  */
/*                                                                           */
/*    Copyright (C) 1996      Roland Wunderling                              */
/*                  1996-2011 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SoPlex is distributed under the terms of the ZIB Academic Licence.       */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SoPlex; see the file COPYING. If not email to soplex@zib.de.  */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <assert.h>
#include <iostream>

// #define EQ_PREF 1000 

#include "spxdefines.h"
#include "spxdantzigpr.h"

namespace soplex
{

int SPxDantzigPR::selectLeave()
{
   assert(thesolver != 0);

   //    const Real* up  = thesolver->ubBound();
   //    const Real* low = thesolver->lbBound();

   Real best = -theeps;
   int  n    = -1;

   for(int i = thesolver->dim() - 1; i >= 0; --i)
   {
      Real x = thesolver->fTest()[i];

      if (x < -theeps)
      {
         // x *= EQ_PREF * (1 + (up[i] == low[i]));
         if (x < best)
         {
            n    = i;
            best = x;
         }
      }
   }
   return n;
}

SPxId SPxDantzigPR::selectEnter()
{
   assert(thesolver != 0);

   // const SPxBasis::Desc&    ds   = thesolver->basis().desc();

   SPxId id;
   int   i;
   Real  best = -theeps;

   for (i = thesolver->dim() - 1; i >= 0; --i) 
   {
      Real x = thesolver->coTest()[i];

      if (x < -theeps)
      {
         // x *= EQ_PREF * (1 + (ds.coStatus(i) == SPxBasis::Desc::P_FREE
         //                || ds.coStatus(i) == SPxBasis::Desc::D_FREE));
         if (x < best)
         {
            id   = thesolver->coId(i);
            best = x;
         }
      }
   }
   for (i = thesolver->coDim() - 1; i >= 0; --i)
   {
      Real x = thesolver->test()[i];

      if (x < -theeps)
      {
         // x *= EQ_PREF * (1 + (ds.status(i) == SPxBasis::Desc::P_FREE
         //                || ds.status(i) == SPxBasis::Desc::D_FREE));
         if (x < best)
         {
            id   = thesolver->id(i);
            best = x;
         }
      }
   }
   return id;
}
} // namespace soplex

//-----------------------------------------------------------------------------
//Emacs Local Variables:
//Emacs mode:c++
//Emacs c-basic-offset:3
//Emacs tab-width:8
//Emacs indent-tabs-mode:nil
//Emacs End:
//-----------------------------------------------------------------------------



