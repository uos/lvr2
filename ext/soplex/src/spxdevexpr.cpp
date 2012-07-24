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

#include "spxdefines.h"
#include "spxdevexpr.h"
#include "message.h"

#define DEVEX_REFINETOL 2.0

namespace soplex
{

void SPxDevexPR::load(SPxSolver* base)
{
   thesolver = base;
   setRep(base->rep());
   assert(isConsistent());
}

bool SPxDevexPR::isConsistent() const
{
#ifdef ENABLE_CONSISTENCY_CHECKS
   if (thesolver != 0)
      if (penalty.dim() != thesolver->coDim()
           || coPenalty.dim() != thesolver->dim())
         return MSGinconsistent("SPxDevexPR");
#endif

   return true;
}

void SPxDevexPR::init(SPxSolver::Type tp)
{
   int i;
   if (tp == SPxSolver::ENTER)
   {
      for (i = penalty.dim(); --i >= 0;)
         penalty[i] = 2;
      for (i = coPenalty.dim(); --i >= 0;)
         coPenalty[i] = 2;
   }
   else
   {
      for (i = coPenalty.dim(); --i >= 0;)
         coPenalty[i] = 1;
   }
   assert(isConsistent());
}

void SPxDevexPR::setType(SPxSolver::Type tp)
{
   init(tp);
   refined = false;
}

/**@todo suspicious: Shouldn't the relation between dim, coDim, Vecs, 
 *       and CoVecs be influenced by the representation ?
 */
void SPxDevexPR::setRep(SPxSolver::Representation)
{
   if (thesolver != 0)
   {
      addedVecs(thesolver->coDim());
      addedCoVecs(thesolver->dim());
      assert(isConsistent());
   }
}

int SPxDevexPR::selectLeave()
{
   int retid;
   Real val;

   retid = selectLeaveX(val, theeps);

   if( retid < 0 && !refined )
   {
      refined = true;
      MSG_INFO3( spxout << "WDEVEX02 trying refinement step..\n"; )
      retid = selectLeaveX(val, theeps/DEVEX_REFINETOL);
   }

   return retid;
}

int SPxDevexPR::selectLeaveX(Real& best, Real feastol, int start, int incr)
{
   Real x;

   const Real* fTest = thesolver->fTest().get_const_ptr();
   const Real* cpen = coPenalty.get_const_ptr();
   Real bstX = 0;
   int bstI = -1;
   int end = coPenalty.dim();

   for (; start < end; start += incr)
   {
      if (fTest[start] < -feastol)
      {
         x = fTest[start] * fTest[start] / cpen[start];
         if (x > bstX)
         {
            bstX = x;
            bstI = start;
            last = cpen[start];
         }
      }
   }
   best = bstX;
   return bstI;
}

void SPxDevexPR::left4(int n, SPxId id)
{
   left4X(n, id, 0, 1);
}

void SPxDevexPR::left4X(int n, const SPxId& id, int start, int incr)
{
   if (id.isValid())
   {
      int i, j;
      Real x;
      const Real* rhoVec = thesolver->fVec().delta().values();
      Real rhov_1 = 1 / rhoVec[n];
      Real beta_q = thesolver->coPvec().delta().length2() * rhov_1 * rhov_1;

#ifndef NDEBUG
      if (fabs(rhoVec[n]) < theeps)
      {
         MSG_ERROR( spxout << "WDEVEX01: rhoVec = "
                           << rhoVec[n] << " with smaller absolute value than theeps = " << theeps << std::endl; )
      }
#endif  // NDEBUG

      //  Update #coPenalty# vector
      const IdxSet& rhoIdx = thesolver->fVec().idx();
      int len = thesolver->fVec().idx().size();
      for (i = len - 1 - start; i >= 0; i -= incr)
      {
         j = rhoIdx.index(i);
         x = rhoVec[j] * rhoVec[j] * beta_q;
         // if(x > coPenalty[j])
         coPenalty[j] += x;
      }

      coPenalty[n] = beta_q;
   }
}

SPxId SPxDevexPR::selectEnter()
{
   SPxId retid;
   Real val;

   retid = selectEnterX(val, theeps);

   if( !retid.isValid() && !refined )
   {
      refined = true;
      MSG_INFO3( spxout << "WDEVEX02 trying refinement step..\n"; )
      retid = selectEnterX(val, theeps/DEVEX_REFINETOL);
   }

   return retid;
}

SPxId SPxDevexPR::selectEnterX(
   Real& best,
   Real feastol,
   int start1,
   int incr1,
   int start2,
   int incr2
   )
{
   Real x;

   const Real* test = thesolver->test().get_const_ptr();
   const Real* cTest = thesolver->coTest().get_const_ptr();
   const Real* cpen = coPenalty.get_const_ptr();
   const Real* pen = penalty.get_const_ptr();
   Real bstX1 = 0;
   Real bstX2 = 0;
   int bstI1 = -1;
   int bstI2 = -1;
   int end1 = coPenalty.dim();
   int end2 = penalty.dim();

   assert(end1 == thesolver->coTest().dim());
   assert(end2 == thesolver->test().dim());

   for (; start1 < end1; start1 += incr1)
   {
      if (cTest[start1] < -feastol)
      {
         x = cTest[start1] * cTest[start1] / cpen[start1];
         if (x > bstX1)
         {
            bstX1 = x;
            bstI1 = start1;
            last = cpen[start1];
         }
      }
   }

   for (; start2 < end2; start2 += incr2)
   {
      if (test[start2] < -feastol)
      {
         x = test[start2] * test[start2] / pen[start2];
         if (x > bstX2)
         {
            bstX2 = x;
            bstI2 = start2;
            last = pen[start2];
         }
      }
   }

   if (bstI2 >= 0)
   {
      best = bstX2;
      return thesolver->id(bstI2);
   }

   if (bstI1 >= 0)
   {
      best = bstX1;
      return thesolver->coId(bstI1);
   }

   SPxId none;
   return none;
}

void SPxDevexPR::entered4(SPxId id, int n)
{
   entered4X(id, n, 0, 1, 0, 1);
}

/**@todo suspicious: the pricer should be informed, that variable id 
    has entered the basis at position n, but the id is not used here 
    (this is true for all pricers)
*/
void SPxDevexPR::entered4X(SPxId /*id*/, int n,
   int start1, int incr1, int start2, int incr2)
{
   if (n >= 0 && n < thesolver->dim())
   {
      const Real* pVec = thesolver->pVec().delta().values();
      const IdxSet& pIdx = thesolver->pVec().idx();
      const Real* coPvec = thesolver->coPvec().delta().values();
      const IdxSet& coPidx = thesolver->coPvec().idx();
      Real xi_p = 1 / thesolver->fVec().delta()[n];
      int i, j;

      assert(thesolver->fVec().delta()[n] > thesolver->epsilon()
              || thesolver->fVec().delta()[n] < -thesolver->epsilon());

      xi_p = xi_p * xi_p * last;

      for (j = coPidx.size() - 1 - start1; j >= 0; j -= incr1)
      {
         i = coPidx.index(j);
         coPenalty[i] += xi_p * coPvec[i] * coPvec[i];
         if (coPenalty[i] <= 1 || coPenalty[i] > 1e+6)
         {
            init(SPxSolver::ENTER);
            return;
         }
      }

      for (j = pIdx.size() - 1 - start2; j >= 0; j -= incr2)
      {
         i = pIdx.index(j);
         penalty[i] += xi_p * pVec[i] * pVec[i];
         if (penalty[i] <= 1 || penalty[i] > 1e+6)
         {
            init(SPxSolver::ENTER);
            return;
         }
      }
   }
}

void SPxDevexPR::addedVecs (int n)
{
   int initval = (thesolver->type() == SPxSolver::ENTER) ? 2 : 1;
   n = penalty.dim();
   penalty.reDim (thesolver->coDim());
   for (int i = penalty.dim()-1; i >= n; --i )
      penalty[i] = initval;
}

void SPxDevexPR::addedCoVecs(int n)
{
   int initval = (thesolver->type() == SPxSolver::ENTER) ? 2 : 1;
   n = coPenalty.dim();
   coPenalty.reDim(thesolver->dim());
   for (int i = coPenalty.dim()-1; i >= n; --i)
      coPenalty[i] = initval;
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
