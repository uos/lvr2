/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the class library                   */
/*       SoPlex --- the Sequential object-oriented simPlex.                  */
/*                                                                           */
/*    Copyright (C) 1997-1999 Roland Wunderling                              */
/*                  1997-2011 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SoPlex is distributed under the terms of the ZIB Academic Licence.       */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SoPlex; see the file COPYING. If not email to soplex@zib.de.  */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

//#define DEBUGGING 1

#include <assert.h>
#include "spxdefines.h"
#include "spxboundflippingrt.h"
#include "sorter.h"
#include "spxsolver.h"
#include "spxout.h"
#include "spxid.h"

namespace soplex
{

#define MINSTAB          1e-5
#define LOWSTAB          1e-10
#define MAX_RELAX_COUNT  2
#define LONGSTEP_FREQ    500
#define MIN_LONGSTEP     1e-2


/** perform necessary bound flips to restore dual feasibility */
void SPxBoundFlippingRT::flipAndUpdate(
   int&                  usedBp              /**< number of bounds that should be flipped */
   )
{
   assert(thesolver->rep() == SPxSolver::COLUMN);

   int skipped;

   updPrimRhs.setup();
   updPrimRhs.reDim(thesolver->dim());
   updPrimVec.reDim(thesolver->dim());
   updPrimRhs.clear();
   updPrimVec.clear();

   skipped = 0;
   for( int i = 0; i < usedBp; ++i )
   {
      int idx;
      idx = breakpoints[i].idx;
      if( idx < 0 )
      {
         ++skipped;
         continue;
      }
      Real tmp;
      Real range;
      Real upper;
      Real lower;
      SPxBasis::Desc::Status stat;
      SPxBasis::Desc& ds = thesolver->basis().desc();
      range = 0;
      if( breakpoints[i].src == PVEC )
      {
         stat = ds.status(idx);
         upper = thesolver->upper(idx);
         lower = thesolver->lower(idx);
         switch( stat )
         {
            case SPxBasis::Desc::P_ON_UPPER :
               ds.status(idx) = SPxBasis::Desc::P_ON_LOWER;
               range = lower - upper;
               tmp = (*thesolver->theLbound)[idx];
               (*thesolver->theLbound)[idx] = (*thesolver->theUbound)[idx];
               (*thesolver->theUbound)[idx] = -tmp;
               break;
            case SPxBasis::Desc::P_ON_LOWER :
               ds.status(idx) = SPxBasis::Desc::P_ON_UPPER;
               range = upper - lower;
               tmp = (*thesolver->theUbound)[idx];
               (*thesolver->theUbound)[idx] = (*thesolver->theLbound)[idx];
               (*thesolver->theLbound)[idx] = -tmp;
               break;
            default :
               ++skipped;
               MSG_WARNING( spxout << "PVEC unexpected status: " << stat
                                   << " index: " << idx
                                   << " val: " << thesolver->pVec()[idx]
                                   << " upd: " << thesolver->pVec().delta()[idx]
                                   << " lower: " << lower
                                   << " upper: " << upper
                                   << " bp.val: " << breakpoints[i].val
                                   << std::endl; )
         }
         MSG_DEBUG( spxout << "PVEC flipped from: " << stat
                           << " index: " << idx
                           << " val: " << thesolver->pVec()[idx]
                           << " upd: " << thesolver->pVec().delta()[idx]
                           << " lower: " << lower
                           << " upper: " << upper
                           << " bp.val: " << breakpoints[i].val
                           << std::endl; )
         assert(fabs(range) < 1e20);
         updPrimRhs.multAdd(range, thesolver->vector(idx));
      }
      else if( breakpoints[i].src == COPVEC )
      {
         stat = ds.coStatus(idx);
         upper = thesolver->rhs(idx);
         lower = thesolver->lhs(idx);
         switch( stat )
         {
            case SPxBasis::Desc::P_ON_UPPER :
               ds.coStatus(idx) = SPxBasis::Desc::P_ON_LOWER;
               range = lower - upper;
               tmp = (*thesolver->theCoLbound)[idx];
               (*thesolver->theCoLbound)[idx] = -(*thesolver->theCoUbound)[idx];
               (*thesolver->theCoUbound)[idx] = tmp;
               break;
            case SPxBasis::Desc::P_ON_LOWER :
               ds.coStatus(idx) = SPxBasis::Desc::P_ON_UPPER;
               range = upper - lower;
               tmp = (*thesolver->theCoUbound)[idx];
               (*thesolver->theCoUbound)[idx] = -(*thesolver->theCoLbound)[idx];
               (*thesolver->theCoLbound)[idx] = tmp;
               break;
            default :
               ++skipped;
               MSG_WARNING( spxout << "COPVEC unexpected status: " << stat
                                   << " index: " << idx
                                   << " val: " << thesolver->coPvec()[idx]
                                   << " upd: " << thesolver->coPvec().delta()[idx]
                                   << " lower: " << lower
                                   << " upper: " << upper
                                   << " bp.val: " << breakpoints[i].val
                                   << std::endl; )
         }
         MSG_DEBUG( spxout << "COPVEC flipped from: " << stat
                           << " index: " << idx
                           << " val: " << thesolver->coPvec()[idx]
                           << " upd: " << thesolver->coPvec().delta()[idx]
                           << " lower: " << lower
                           << " upper: " << upper
                           << " bp.val: " << breakpoints[i].val
                           << std::endl; )
         assert(fabs(range) < 1e20);
         updPrimRhs.setValue(idx, updPrimRhs[idx] - range);
      }
   }
   usedBp -= skipped;
   if( usedBp > 0 )
   {
      thesolver->primRhs -= updPrimRhs;
      thesolver->setup4solve2(&updPrimVec, &updPrimRhs);
   }

   return;
}

/** store all available pivots/breakpoints in an array (positive pivot search direction) */
void SPxBoundFlippingRT::collectBreakpointsMax(
   int&                  nBp,                /**< number of found breakpoints so far */
   int&                  minIdx,             /**< index to current minimal breakpoint */
   const int*            idx,                /**< pointer to indices of current vector */
   int                   nnz,                /**< number of nonzeros in current vector */
   const Real*           upd,                /**< pointer to update values of current vector */
   const Real*           vec,                /**< pointer to values of current vector */
   const Real*           upp,                /**< pointer to upper bound/rhs of current vector */
   const Real*           low,                /**< pointer to lower bound/lhs of current vector */
   BreakpointSource      src                 /**< type of vector (pVec or coPvec)*/
   )
{
   Real minVal;
   Real curVal;
   const int* last;

   minVal = ( nBp == 0 ) ? infinity : breakpoints[minIdx].val;

   last = idx + nnz;

   for( ; idx < last; ++idx )
   {
      int i = *idx;
      Real x = upd[i];
      if( x > epsilon )
      {
         if( upp[i] < infinity )
         {
            Real y = upp[i] - vec[i];
            curVal = (y <= 0) ? delta / x : (y + delta) / x;
            assert(curVal > 0);

            breakpoints[nBp].idx = i;
            breakpoints[nBp].src = src;
            breakpoints[nBp].val = curVal;

            if( curVal < minVal )
            {
               minVal = curVal;
               minIdx = nBp;
            }

            nBp++;
         }
      }
      else if( x < -epsilon )
      {
         if (low[i] > -infinity)
         {
            Real y = low[i] - vec[i];
            curVal = (y >= 0) ? -delta / x : (y - delta) / x;
            assert(curVal > 0);

            breakpoints[nBp].idx = i;
            breakpoints[nBp].src = src;
            breakpoints[nBp].val = curVal;

            if( curVal < minVal )
            {
               minVal = curVal;
               minIdx = nBp;
            }

            nBp++;
         }
      }
      if( nBp >= breakpoints.size() )
         breakpoints.reSize(nBp * 2);
   }

   return;
}

/** store all available pivots/breakpoints in an array (negative pivot search direction) */
void SPxBoundFlippingRT::collectBreakpointsMin(
   int&                  nBp,                /**< number of found breakpoints so far */
   int&                  minIdx,             /**< index to current minimal breakpoint */
   const int*            idx,                /**< pointer to indices of current vector */
   int                   nnz,                /**< number of nonzeros in current vector */
   const Real*           upd,                /**< pointer to update values of current vector */
   const Real*           vec,                /**< pointer to values of current vector */
   const Real*           upp,                /**< pointer to upper bound/rhs of current vector */
   const Real*           low,                /**< pointer to lower bound/lhs of current vector */
   BreakpointSource      src                 /**< type of vector (pVec or coPvec)*/
   )
{
   Real minVal;
   Real curVal;
   const int* last;

   minVal = ( nBp == 0 ) ? infinity : breakpoints[minIdx].val;

   last = idx + nnz;

   for( ; idx < last; ++idx )
   {
      int i = *idx;
      Real x = upd[i];
      if( x > epsilon )
      {
         if( low[i] > -infinity )
         {
            Real y = low[i] - vec[i];

            curVal = (y >= 0) ? delta / x : (delta - y) / x;
            assert(curVal > 0);

            breakpoints[nBp].idx = i;
            breakpoints[nBp].src = src;
            breakpoints[nBp].val = curVal;

            if( curVal < minVal )
            {
               minVal = curVal;
               minIdx = nBp;
            }

            nBp++;
         }
      }
      else if( x < -epsilon )
      {
         if (upp[i] < infinity)
         {
            Real y = upp[i] - vec[i];
            curVal = (y <= 0) ? -delta / x : -(y + delta) / x;
            assert(curVal > 0);

            breakpoints[nBp].idx = i;
            breakpoints[nBp].src = src;
            breakpoints[nBp].val = curVal;

            if( curVal < minVal )
            {
               minVal = curVal;
               minIdx = nBp;
            }

            nBp++;
         }
      }
      if( nBp >= breakpoints.size() )
         breakpoints.reSize(nBp * 2);
   }
   return;
}

/** determine entering variable */
SPxId SPxBoundFlippingRT::selectEnter(
   Real&                 val,
   int                   leaveIdx
   )
{
   assert( m_type == SPxSolver::LEAVE );
   assert(thesolver->boundflips == 0);

   // reset the history and try again to do some long steps
   if( thesolver->leaveCount % LONGSTEP_FREQ == 0 )
      flipPotential = 1;
   if( !enableLongsteps || thesolver->rep() == SPxSolver::ROW || flipPotential < 0.01 )
   {
      return SPxFastRT::selectEnter(val, leaveIdx);
   }
   const Real*  pvec = thesolver->pVec().get_const_ptr();
   const Real*  pupd = thesolver->pVec().delta().values();
   const int*   pidx = thesolver->pVec().delta().indexMem();
   int          pupdnnz = thesolver->pVec().delta().size();
   const Real*  lpb  = thesolver->lpBound().get_const_ptr();
   const Real*  upb  = thesolver->upBound().get_const_ptr();

   const Real*  cvec = thesolver->coPvec().get_const_ptr();
   const Real*  cupd = thesolver->coPvec().delta().values();
   const int*   cidx = thesolver->coPvec().delta().indexMem();
   int          cupdnnz = thesolver->coPvec().delta().size();
   const Real*  lcb  = thesolver->lcBound().get_const_ptr();
   const Real*  ucb  = thesolver->ucBound().get_const_ptr();

   resetTols();

   Real max;

   // index in breakpoint array of minimal value (i.e. choice of normal RT)
   int minIdx;

   // temporary breakpoint data structure to make swaps possible
   Breakpoint tmp;

   // most stable pivot value in candidate set
   Real moststable;

   // initialize invalid enterId
   SPxId enterId;

   // slope of objective function improvement
   Real slope;

   // number of found breakpoints
   int nBp;

   // number of latest skipable breakpoint
   int usedBp;

   Real degeneps;
   Real stab;
   bool instable;

   max = val;
   val = 0.0;
   moststable = 0.0;
   nBp = 0;
   minIdx = -1;

   // get breakpoints and and determine the index of the minimal value
   if( max > 0 )
   {
      collectBreakpointsMax(nBp, minIdx, pidx, pupdnnz, pupd, pvec, upb, lpb, PVEC);
      collectBreakpointsMax(nBp, minIdx, cidx, cupdnnz, cupd, cvec, ucb, lcb, COPVEC);
   }
   else
   {
      collectBreakpointsMin(nBp, minIdx, pidx, pupdnnz, pupd, pvec, upb, lpb, PVEC);
      collectBreakpointsMin(nBp, minIdx, cidx, cupdnnz, cupd, cvec, ucb, lcb, COPVEC);
   }

   if( nBp == 0 )
      return enterId;

   assert(minIdx >= 0);

   // swap smallest breakpoint to the front to skip the sorting phase if no bound flip is possible
   tmp = breakpoints[minIdx];
   breakpoints[minIdx] = breakpoints[0];
   breakpoints[0] = tmp;

   // compute initial slope
   slope = fabs(thesolver->fTest()[leaveIdx]);
   if( slope == 0 )
   {
      // this may only happen if SoPlex decides to make an instable pivot
      assert(thesolver->instableLeaveNum >= 0);
      // restore original slope
      slope = fabs(thesolver->instableLeaveVal);
   }

   // set up structures for the quicksort implementation
   BreakpointCompare compare;
   compare.entry = breakpoints.get_const_ptr();

   // pointer to end of sorted part of breakpoints
   int sorted = 0;
   // minimum number of entries that are supposed to be sorted by partial sort
   int sortsize = 4;

   // get all skipable breakpoints
   for( usedBp = 0; usedBp < nBp && slope > 0; ++usedBp)
   {
      // sort breakpoints only partially to save time
      if( usedBp > sorted )
      {
         sorted = sorter_qsortPart(breakpoints.get_ptr(), compare, sorted + 1, nBp, sortsize);
      }
      int i = breakpoints[usedBp].idx;
      // compute new slope
      if( breakpoints[usedBp].src == PVEC )
      {
         if( thesolver->isBasic(i) )
         {
            // mark basic indices
            breakpoints[usedBp].idx = -1;
            thesolver->pVec().delta().clearIdx(i);
         }
         else
         {
            Real absupd = fabs(pupd[i]);
            slope -= (thesolver->upper(i) - thesolver->lower(i)) * absupd;
            if( absupd > moststable )
               moststable = absupd;
         }
      }
      else
      {
         assert(breakpoints[usedBp].src == COPVEC);
         if( thesolver->isCoBasic(i) )
         {
            // mark basic indices
            breakpoints[usedBp].idx = -1;
            thesolver->coPvec().delta().clearIdx(i);
         }
         else
         {
            Real absupd = fabs(cupd[i]);
            slope -= (thesolver->rhs(i) - thesolver->lhs(i)) * absupd;
            if( absupd > moststable )
               moststable = absupd;
         }
      }
   }
   --usedBp;
   assert(usedBp >= 0);

   MSG_DEBUG( spxout << "DLSTEP01 "
                     << thesolver->basis().iteration()
                     << ": number of flip candidates: "
                     << usedBp
                     << std::endl; )

   // check for unboundedness/infeasibility
   if( slope > epsilon && usedBp >= nBp - 1 )
   {
      MSG_DEBUG( spxout << "DLSTEP02 "
                        << thesolver->basis().iteration()
                        << ": unboundedness in ratio test"
                        << std::endl; )
      flipPotential *= 0.95;
      val = max;
      return SPxFastRT::selectEnter(val, leaveIdx);
   }

   // do not make long steps if the gain in the dual objective is too small, except to avoid degenerate steps
   if( usedBp > 0 && breakpoints[0].val > epsilon && breakpoints[usedBp].val - breakpoints[0].val < MIN_LONGSTEP * breakpoints[0].val )
   {
      MSG_DEBUG( spxout << "DLSTEP03 "
                        << thesolver->basis().iteration()
                        << ": bound flip gain is too small"
                        << std::endl; )
      usedBp = 0;

      // ensure that the first breakpoint is nonbasic
      while( breakpoints[usedBp].idx < 0 && usedBp < nBp )
         ++usedBp;
      // @todo make sure that the selected pivot element is stable
   }

   // scan pivot candidates from back to front and stop as soon as a good one is found
   degeneps = delta / moststable;  /* as in SPxFastRT */
   instable = thesolver->instableLeave;
   assert(!instable || thesolver->instableLeaveNum >= 0);
   stab = instable ? LOWSTAB : SPxFastRT::minStability(moststable);
   // @todo select the moststable pivot or one that is comparably stable
   // stab = moststable * 1e-2;

   while( usedBp >= 0 )
   {
      int idx = breakpoints[usedBp].idx;

      // skip basic variables
      if( idx < 0 )
      {
         --usedBp;
         continue;
      }
      if( max > 0 )
      {
         if( breakpoints[usedBp].src == PVEC )
         {
            thesolver->pVec()[idx] = thesolver->vector(idx) * thesolver->coPvec();
            Real x = pupd[idx];
            // skip breakpoint if it is too small
            if( fabs(x) < stab )
            {
               --usedBp;
               continue;
            }
            enterId = thesolver->id(idx);
            val = (x > 0.0) ? upb[idx] : lpb[idx];
            val = (val - pvec[idx]) / x;
            if( upb[idx] == lpb[idx] )
            {
               val = 0.0;
               if( pvec[idx] > upb[idx] )
                  thesolver->theShift += pvec[idx] - upb[idx];
               else
                  thesolver->theShift += lpb[idx] - pvec[idx];
               thesolver->upBound()[idx] = thesolver->lpBound()[idx] = pvec[idx];
            }
            else if( val < -degeneps )
            {
               val = 0.0;
               if( x > 0.0 )
                  thesolver->shiftUPbound(idx, pvec[idx]);
               else
                  thesolver->shiftLPbound(idx, pvec[idx]);
            }
         }
         else // breakpoints[usedBp].src == COPVEC
         {
            Real x = cupd[idx];
            if( fabs(x) < stab )
            {
               --usedBp;
               continue;
            }
            enterId = thesolver->coId(idx);
            val = (x > 0.0) ? ucb[idx] : lcb[idx];
            val = (val - cvec[idx]) / x;
            if( ucb[idx] == lcb[idx] )
            {
               val = 0.0;
               if( cvec[idx] > ucb[idx] )
                  thesolver->theShift += cvec[idx] - ucb[idx];
               else
                  thesolver->theShift += lcb[idx] - cvec[idx];
               thesolver->ucBound()[idx] = thesolver->lcBound()[idx] = cvec[idx];
            }
            else if( val < -degeneps )
            {
               val = 0.0;
               if( x > 0.0 )
                  thesolver->shiftUCbound(idx, cvec[idx]);
               else
                  thesolver->shiftLCbound(idx, cvec[idx]);
            }
         }
      }
      else // (max < 0)
      {
         if( breakpoints[usedBp].src == PVEC )
         {
            Real x;

            thesolver->pVec()[idx] = thesolver->vector(idx) * thesolver->coPvec();
            x = pupd[idx];
            if( fabs(x) < stab )
            {
               --usedBp;
               continue;
            }

            enterId = thesolver->id(idx);
            val = (x < 0.0) ? upb[idx] : lpb[idx];
            val = (val - pvec[idx]) / x;
            if( upb[idx] == lpb[idx] )
            {
               val = 0.0;
               if( pvec[idx] > upb[idx] )
                  thesolver->theShift += pvec[idx] - upb[idx];
               else
                  thesolver->theShift += lpb[idx] - pvec[idx];

               thesolver->upBound()[idx] = thesolver->lpBound()[idx] = pvec[idx];
            }
            else if( val > degeneps )
            {
               val = 0.0;
               if( x > 0.0 )
                  thesolver->shiftLPbound(idx, pvec[idx]);
               else
                  thesolver->shiftUPbound(idx, pvec[idx]);
            }
         }
         else // breakpoints[usedBp].src == COPVEC
         {
            Real x = cupd[idx];
            if( fabs(x) < stab )
            {
               --usedBp;
               continue;
            }
            enterId = thesolver->coId(idx);
            val = (x < 0.0) ? ucb[idx] : lcb[idx];
            val = (val - cvec[idx]) / x;
            if( ucb[idx] == lcb[idx] )
            {
               val = 0.0;
               if( cvec[idx] > ucb[idx] )
                  thesolver->theShift += cvec[idx] - ucb[idx];
               else
                  thesolver->theShift += lcb[idx] - cvec[idx];

               thesolver->ucBound()[idx] = thesolver->lcBound()[idx] = cvec[idx];
            }
            else if( val > degeneps )
            {
               val = 0.0;
               if( x > 0.0 )
                  thesolver->shiftLCbound(idx, cvec[idx]);
               else
                  thesolver->shiftUCbound(idx, cvec[idx]);
            }
         }
      }
      break;
   }

   if( !enterId.isValid())
   {
      assert(usedBp < 0);
      if( relax_count < MAX_RELAX_COUNT )
      {
         MSG_DEBUG( spxout << "DLSTEP04 "
                           << thesolver->basis().iteration()
                           << ": no valid enterId found - relaxing..."
                           << std::endl; )
         relax();
         ++relax_count;
         // restore original value
         val = max;
         return SPxBoundFlippingRT::selectEnter(val, leaveIdx);
      }
      else
      {
         MSG_DEBUG( spxout << "DLSTEP05 "
                           << thesolver->basis().iteration()
                           << "no valid enterId found - breaking..."
                           << std::endl; )
         return enterId;
      }
   }
   else
   {
      assert(usedBp >= 0);
      relax_count = 0;
      tighten();
   }
   if( usedBp > 0 )
      flipAndUpdate(usedBp);

   thesolver->boundflips = usedBp;
   // estimate wether long steps may be possible in future iterations
   flipPotential *= (usedBp + 0.95);

   MSG_DEBUG( spxout << "DLSTEP06 "
                     << thesolver->basis().iteration()
                     << ": selected Id: "
                     << enterId
                     << std::endl; )
   return enterId;
}

int SPxBoundFlippingRT::selectLeave(
   Real&                 max,
   SPxId                 enterId
   )
{
   return SPxFastRT::selectLeave(max, enterId);
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
