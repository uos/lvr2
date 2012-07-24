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

//#define DEBUGGING 1

#include <assert.h>
#include <iostream>

#include "spxdefines.h"
#include "spxsolver.h"
#include "spxpricer.h"
#include "spxratiotester.h"
#include "spxdefaultrt.h"
#include "spxstarter.h"
#include "spxout.h"
#include "exceptions.h"

#define MAXCYCLES 400
#define MAXSTALLS 10000
#define MAXSTALLRECOVERS 10

namespace soplex
{
/// Interval for displaying iteration information.
long iterationInterval = 100;


bool SPxSolver::precisionReached(Real& newDelta) const
{
   Real maxViolRedCost;
   Real sumViolRedCost;
   Real maxViolBounds;
   Real sumViolBounds;
   Real maxViolConst;
   Real sumViolConst;

   qualRedCostViolation(maxViolRedCost, sumViolRedCost);
   qualBoundViolation(maxViolBounds, sumViolBounds);
   qualConstraintViolation(maxViolConst, sumViolConst);

   // is the solution good enough ?
   bool reached = maxViolRedCost < delta() && maxViolBounds < delta() && maxViolConst < delta();

   if (!reached)
   {
      newDelta = thepricer->epsilon() / 10.0;

      MSG_INFO3( spxout << "ISOLVE71 "
                           << "Precision not reached: Pricer delta= " 
                           << thepricer->epsilon() 
                           << " new delta= " << newDelta
                           << std::endl
                           << " maxViolRedCost= " << maxViolRedCost
                           << " maxViolBounds= " << maxViolBounds
                           << " maxViolConst= " << maxViolConst
                           << std::endl
                           << " sumViolRedCost= " << sumViolRedCost
                           << " sumViolBounds= " << sumViolBounds
                           << " sumViolConst= " << sumViolConst
                           << std::endl; );
   }
   return reached;
}

/**@todo After solve() returned, the algorithm type may have changed.
 *       This may be a problem if solve() is called again.
 * @todo The errors at the beginning do not set m_status. On the other
 *       hand none of the routines that change for example the pricer
 *       changes the status.
 */
SPxSolver::Status SPxSolver::solve()
{
   METHOD( "SPxSolver::solve()" );

   SPxId enterId;
   int   leaveNum;
   int   loopCount = 0;
   Real  minDelta;
   Real  maxDelta;
   Real  newDelta;
   Real  minShift = infinity;
   int   cycleCount = 0;

   /* store the last (primal or dual) feasible objective value to recover/abort in case of stalling */
   Real  stallRefValue;
   Real  stallRefShift;
   int   stallRefIter;
   int   stallNumRecovers;

   if (dim() <= 0 && coDim() <= 0) // no problem loaded
   {
      m_status = NO_PROBLEM;
      throw SPxStatusException("XSOLVE01 No Problem loaded");
   }

   if (slinSolver() == 0) // linear system solver is required.
   {
      m_status = NO_SOLVER;
      throw SPxStatusException("XSOLVE02 No Solver loaded");
   }
   if (thepricer == 0) // pricer is required.
   {
      m_status = NO_PRICER;
      throw SPxStatusException("XSOLVE03 No Pricer loaded");
   }
   if (theratiotester == 0) // ratiotester is required.
   {
      m_status = NO_RATIOTESTER;
      throw SPxStatusException("XSOLVE04 No RatioTester loaded");
   }
   theTime.reset();
   theTime.start();

   m_numCycle = 0;
   iterCount  = 0;
   if (!isInitialized())
   {
      /*
      if(SPxBasis::status() <= NO_PROBLEM)
          SPxBasis::load(this);
       */
      /**@todo != REGULAR is not enough. Also OPTIMAL/DUAL/PRIMAL should
       * be tested and acted accordingly.
       */
      if (thestarter != 0 && status() != REGULAR)  // no basis and no starter.
         thestarter->generate(*this);              // generate start basis.

      init();

      // Inna/Tobi: init might fail, if the basis is singular
      if( !isInitialized() )
      {
         assert(SPxBasis::status() == SPxBasis::SINGULAR);
         m_status = UNKNOWN;
         return status();
      }
   }
   maxDelta = delta();
   minDelta = delta() * 1e-2;

   //setType(type());

   if (!matrixIsSetup)
      SPxBasis::load(this);

   //factorized = false;

   assert(thepricer->solver()      == this);
   assert(theratiotester->solver() == this);

   // maybe this should be done in init() ?
   thepricer->setType(type());
   theratiotester->setType(type());

   MSG_INFO3(
      spxout << "ISOLVE72 starting value = " << value() << std::endl;
      spxout << "ISOLVE73 starting shift = " << shift() << std::endl; 
   )
   MSG_DEBUG( desc().dump(); )

   if (SPxBasis::status() == SPxBasis::OPTIMAL)
      setBasisStatus(SPxBasis::REGULAR);

   m_status   = RUNNING;
   bool stop  = terminate();
   leaveCount = 0;
   enterCount = 0;
   boundflips = 0;
   totalboundflips = 0;

   stallNumRecovers = 0;

   // save the current basis and tolerance; if we run into a singular basis, we will restore it and try
   // with tighter tolerance
   const Real origtol = maxDelta;
   const SPxBasis::Desc origdesc = desc();
   const SPxSolver::Type origtype = theType;
   bool tightened = false;

   while (!stop)
   {
      try
      {

      if (type() == ENTER)
      {
         int enterCycleCount = 0;

         stallRefIter = iteration()-1;
         stallRefShift = shift();
         stallRefValue = value();

         thepricer->setEpsilon(maxDelta);

         do
         {
            MSG_INFO3(
               if( iteration() % iterationInterval == 0 )
                  spxout << "ISOLVE74 Enter iteration: " << iteration()
                         << ", Value = " << value()
                         << ", Shift = " << shift() << std::endl;
            )
            enterId = thepricer->selectEnter();

            if (!enterId.isValid())
            {
               // we are not infeasible and have no shift
               if (  shift() <= epsilon()
                  && ( SPxBasis::status() == SPxBasis::REGULAR 
                     || SPxBasis::status() == SPxBasis::DUAL 
                     || SPxBasis::status() == SPxBasis::PRIMAL))
               {
                  // is the solution good enough ?
                  // max three times reduced
                  if ((thepricer->epsilon() > minDelta) && !precisionReached(newDelta))
                  {  // no!
                     // we reduce delta(). Note that if the pricer does not find a candiate
                     // with the reduced delta, we quit, regardless of the violations.
                     if (newDelta < minDelta)
                        newDelta = minDelta;

                     thepricer->setEpsilon(newDelta);

                     MSG_INFO2( spxout << "ISOLVE75 Setting delta= " 
                                          << thepricer->epsilon() 
                                          << std::endl; )
                  }
                  // solution seems good, no check whether we are precise enough
                  else if (lastUpdate() == 0)
                     break;
                  // We have an iterationlimit and everything looks good? Then stop!
                  // 6 is just a number picked.
                  else if (maxIters > 0 && lastUpdate() < 6)
                     break;
               }
               MSG_INFO3( spxout << "ISOLVE76 solve(enter) triggers refactorization" << std::endl; )

               // We better refactor to make sure the solution is ok.
               // BH 2005-12-15: For some reason we must do this even if lastUpdate() == 0,
               // otherwise something goes wrong, e.g. in instances of the siemens test set.
               factorize();
               // Inna/Tobi: if the factorization was found out to be singular, we have to quit
               if (SPxBasis::status() < SPxBasis::REGULAR)
               {
                  MSG_ERROR( spxout << "ESOLVE09 something wrong with factorization, Basis status: " << SPxBasis::status() << std::endl; )
                  stop = true;
                  break;
               }

               enterId = thepricer->selectEnter();

               if (!enterId.isValid())
                  break;
            }

            /* check if we have iterations left */
            if (maxIters >= 0 && iterations() >= maxIters)
            {
               MSG_INFO2( spxout << "ISOLVE53e Maximum number of iterations (" << maxIters
                                 << ") reached" << std::endl; )
               m_status = ABORT_ITER;
               stop = true;
               break;
            }

            enter(enterId);
            assert((testBounds(), 1));
            thepricer->entered4(lastEntered(), lastIndex());
            stop = terminate();
            clearUpdateVecs();
            if (lastIndex() >= 0)
            {
               enterCount++;
               enterCycleCount = 0;
            }
            else
            {
               enterCycleCount++;
               if( enterCycleCount > MAXCYCLES )
               {
                  MSG_INFO2( spxout << "ISOLVE77 Abort solving due to cycling in "
                                       << "entering algorithm" << std::endl; );
                  m_status = ABORT_CYCLING;
                  stop = true;
               }
            }

            /* check every MAXSTALLS iterations whether shift and objective value have not changed */
            if( (iteration() - stallRefIter) % MAXSTALLS == 0 )
            {
               if( fabs(value() - stallRefValue) <= epsilon() && fabs(shift() - stallRefShift) <= epsilon() )
               {
                  if( stallNumRecovers < MAXSTALLRECOVERS )
                  {
                     /* try to recover by unshifting/switching algorithm up to MAXSTALLRECOVERS times (just a number picked) */
                     MSG_INFO3( spxout << "ISOLVE21 Stalling detected - trying to recover by switching to LEAVING algorithm." << std::endl; )

                     ++stallNumRecovers;
                     break;
                  }
                  else
                  {
                     /* giving up */
                     MSG_INFO2( spxout << "ISOLVE22 Abort solving due to stalling in entering algorithm." << std::endl; );

                     m_status = ABORT_CYCLING;
                     stop = true;
                  }
               }
               else
               {
                  /* merely update reference values */
                  stallRefIter = iteration()-1;
                  stallRefShift = shift();
                  stallRefValue = value();
               }
            }

            //@ assert(isConsistent());
         }
         while (!stop);

         MSG_INFO3(
            spxout << "ISOLVE78 Enter finished. iteration: " << iteration() 
                   << ", value: " << value()
                   << ", shift: " << shift()
                   << ", epsilon: " << epsilon()
                   << ", delta: " << delta()
                   << std::endl
                   << "ISOLVE56 stop: " << stop
                   << ", basis status: " << SPxBasis::status() << " (" << int(SPxBasis::status()) << ")"
                   << ", solver status: " << m_status << " (" << int(m_status) << ")" << std::endl;
         )

         if (!stop)
         {
            if (shift() <= epsilon())
            {
               // factorize();
               unShift();

               MSG_INFO3(
                  spxout << "ISOLVE79 maxInfeas: " << maxInfeas()
                         << ", shift: " << shift()
                         << ", delta: " << delta() << std::endl;
               )

               if (maxInfeas() + shift() <= delta())
               {
                  setBasisStatus(SPxBasis::OPTIMAL);
                  m_status = OPTIMAL;
                  break;
               }
            }
            setType(LEAVE);
            init();
            thepricer->setType(type());
            theratiotester->setType(type());
         }
      }
      else
      {
         assert(type() == LEAVE);
         
         int leaveCycleCount = 0;

         instableLeaveNum = -1;
         instableLeave = false;

         stallRefIter = iteration()-1;
         stallRefShift = shift();
         stallRefValue = value();

         thepricer->setEpsilon(maxDelta);

         do
         {
            MSG_INFO3(
               if( iteration() % iterationInterval == 0 )
                  spxout << "ISOLVE80 Leave Iteration: " << iteration()
                         << ", Value = " << value()
                         << ", Shift = " << shift() << std::endl;
            )
            
            leaveNum = thepricer->selectLeave();

            if (leaveNum < 0 && instableLeaveNum >= 0)
            {
               /* no leaving variable was found, but because of instableLeaveNum >= 0 we know
                  that this is due to the scaling of theCoTest[...]. Thus, we use 
                  instableLeaveNum and SPxFastRT::selectEnter shall accept even an instable
                  entering variable. */
               MSG_INFO3(
                  spxout << "ISOLVE98 Trying instable leave iteration" << std::endl;
               )
            
               leaveNum = instableLeaveNum;
               instableLeave = true;
            }
            else
            {
               instableLeave = false;
            }

            if (leaveNum < 0)
            {
               // we are not infeasible and have no shift
               if (  shift() <= epsilon()
                  && (  SPxBasis::status() == SPxBasis::REGULAR 
                     || SPxBasis::status() == SPxBasis::DUAL 
                     || SPxBasis::status() == SPxBasis::PRIMAL))
               {
                  // is the solution good enough ?
                  // max three times reduced
                  if ((thepricer->epsilon() > minDelta) && !precisionReached(newDelta))
                  {  // no
                     // we reduce delta(). Note that if the pricer does not find a candiate
                     // with the reduced delta, we quit, regardless of the violations.
                     if (newDelta < minDelta)
                        newDelta = minDelta;

                     thepricer->setEpsilon(newDelta);

                     MSG_INFO2( spxout << "ISOLVE81 Setting delta= " 
                                          << thepricer->epsilon() 
                                          << std::endl; );
                  }
                  // solution seems good, no check whether we are precise enough
                  else if (lastUpdate() == 0)
                     break;
                  // We have an iteration limit and everything looks good? Then stop!
                  // 6 is just a number picked.
                  else if (maxIters > 0 && lastUpdate() < 6)
                     break;
               }
               MSG_INFO3( spxout << "ISOLVE82 solve(leave) triggers refactorization" << std::endl; )

               // We better refactor to make sure the solution is ok.
               // BH 2005-12-15: For some reason we must do this even if lastUpdate() == 0,
               // otherwise something goes wrong, e.g. in instances of the siemens test set.
               factorize();
               // Inna/Tobi: if the factorization was found out to be singular, we have to quit
               if (SPxBasis::status() < SPxBasis::REGULAR)
               {
                  MSG_ERROR( spxout << "ESOLVE10 something wrong with factorization, Basis status: " << SPxBasis::status() << std::endl; )
                  stop = true;
                  break;
               }

               leaveNum = thepricer->selectLeave();

               if (leaveNum < 0)
                  break;
            }

            /* check if we have iterations left */
            if (maxIters >= 0 && iterations() >= maxIters)
            {
               MSG_INFO2( spxout << "ISOLVE53l Maximum number of iterations (" << maxIters
                                 << ") reached" << std::endl; )
               m_status = ABORT_ITER;
               stop = true;
               break;
            }

            leave(leaveNum);
            assert((testBounds(), 1));
            thepricer->left4(lastIndex(), lastLeft());
            stop = terminate();
            clearUpdateVecs();
            if( lastEntered().isValid() )
            {
               leaveCount++;
               leaveCycleCount = 0;
            }
            else
            {
               leaveCycleCount++;
               if( leaveCycleCount > MAXCYCLES )
               {
                  MSG_INFO2( spxout << "ISOLVE83 Abort solving due to cycling in leaving algorithm" << std::endl; );
                  m_status = ABORT_CYCLING;
                  stop = true;
               }
            }

            /* check every MAXSTALLS iterations whether shift and objective value have not changed */
            if( (iteration() - stallRefIter) % MAXSTALLS == 0 )
            {
               if( fabs(value() - stallRefValue) <= epsilon() && fabs(shift() - stallRefShift) <= epsilon() )
               {
                  if( stallNumRecovers < MAXSTALLRECOVERS )
                  {
                     /* try to recover by switching algorithm up to MAXSTALLRECOVERS times */
                     MSG_INFO3( spxout << "ISOLVE24 Stalling detected - trying to recover by switching to ENTERING algorithm." << std::endl; )

                     ++stallNumRecovers;
                     break;
                  }
                  else
                  {
                     /* giving up */
                     MSG_INFO2( spxout << "ISOLVE25 Abort solving due to stalling in leaving algorithm" << std::endl; );

                     m_status = ABORT_CYCLING;
                     stop = true;
                  }
               }
               else
               {
                  /* merely update reference values */
                  stallRefIter = iteration()-1;
                  stallRefShift = shift();
                  stallRefValue = value();
               }
            }

            //@ assert(isConsistent());
         }
         while (!stop);

         MSG_INFO3(
            spxout << "ISOLVE84 Leave finished. iteration: " << iteration() 
                   << ", value: " << value()
                   << ", shift: " << shift()
                   << ", epsilon: " << epsilon()
                   << ", delta: " << delta()
                   << std::endl
                   << "ISOLVE57 stop: " << stop
                   << ", basis status: " << SPxBasis::status() << " (" << int(SPxBasis::status()) << ")"
                   << ", solver status: " << m_status << " (" << int(m_status) << ")" << std::endl;
         )

         if (!stop)
         {
            if( shift() < minShift )
            {
               minShift = shift();
               cycleCount = 0;
            }
            else
            {
               cycleCount++;
               if( cycleCount > MAXCYCLES )
               {
                  m_status = ABORT_CYCLING;
                  stop = true;
                  //MSG_INFO2( spxout << "ISOLVE85 Abort solving due to cycling" << std::endl; )
                  throw SPxStatusException("XSOLVE13 Abort solving due to cycling");
               }
               MSG_INFO3(
                  spxout << "ISOLVE86 maxInfeas: " << maxInfeas()
                         << ", shift: " << shift()
                         << ", delta: " << delta()
                         << ", cycle count: " << cycleCount << std::endl;
               )
            }

            if (shift() <= epsilon())
            {
               cycleCount = 0;
               // factorize();
               unShift();

               MSG_INFO3(
                  spxout << "ISOLVE87 maxInfeas: " << maxInfeas()
                         << ", shift: " << shift()
                         << ", delta: " << delta() << std::endl;
               )

               // We stop if we are indeed optimal, or if we have already been
               // two times at this place. In this case it seems futile to
               // continue.
               if (maxInfeas() + shift() <= delta() || loopCount >= 2)
               {
                  setBasisStatus(SPxBasis::OPTIMAL);
                  m_status = OPTIMAL;
                  break;
               }
               loopCount++;
            }
            setType(ENTER);
            init();
            thepricer->setType(type());
            theratiotester->setType(type());
         }
      }
      assert(m_status != SINGULAR);

      }
      catch( SPxException E )
      {
         // if we stopped due to a singular basis, we reload the original basis and try again with tighter
         // tolerance (only once)
         if (m_status == SINGULAR && !tightened)
         {
            Real newtol = 0.001*origtol;

            MSG_INFO2( spxout << "ISOLVE26 basis singular: reloading basis and solving with tighter tolerance " << newtol << std::endl; )

            // tighten tolerances (pricer tolerance automatically set during solve loop)
            setDelta(newtol);
            maxDelta = newtol;
            minDelta = newtol * 1e-2;

            // return to original algorithm type
            if (type() != origtype)
               setType(origtype);

            // load original basis
            int niters = iterations();
            loadBasis(origdesc);

            // remember iteration count
            iterCount = niters;

            // try initializing basis (might fail if starting basis was already singular)
            try
            {
               init();
            }
            catch( SPxException Ex )
            {
               MSG_INFO2( spxout << "ISOLVE27 reloaded basis singular, resetting original tolerances" << std::endl; )

               thepricer->setEpsilon(origtol);
               setDelta(origtol);

               throw Ex;
            }

            // reset status and counters
            m_status = RUNNING;
            m_numCycle = 0;
            leaveCount = 0;
            enterCount = 0;
            stallNumRecovers = 0;

            // continue
            stop = false;
            tightened = true;
         }
         else
         {
            // reset tolerance to its original value
            if (tightened)
            {
               thepricer->setEpsilon(origtol);
               setDelta(origtol);
            }

            throw E;
         }
      }
   }

   // reset tolerance to its original value
   if (tightened)
   {
      thepricer->setEpsilon(origtol);
      setDelta(origtol);
   }

   assert(delta() == origtol);

   theTime.stop();
   theCumulativeTime += time();

   if (m_status == RUNNING)
   {
      m_status = ERROR;
      throw SPxStatusException("XSOLVE05 Status is still RUNNING when it shouldn't be");
   }

   MSG_INFO1(
      spxout << "ISOLVE02 Finished solving (status=" << status()
             << ", iters=" << iterCount
             << ", leave=" << leaveCount
             << ", enter=" << enterCount
             << ", flips=" << totalboundflips;
      if( status() == OPTIMAL )
         spxout << ", objValue=" << value();
      spxout << ")" << std::endl;
   )

#ifdef ENABLE_ADDITIONAL_CHECKS
   /* check if solution is really feasible */
   if( status() == OPTIMAL )
   {
      int     c;
      Real    val;
      DVector sol( nCols() );

      getPrimal( sol );

      for(int row = 0; row < nRows(); ++row )
      {
         const SVector& rowvec = rowVector( row );
         val = 0.0;         
         for( c = 0; c < rowvec.size(); ++c )
            val += rowvec.value( c ) * sol[rowvec.index( c )];

         if( LT( val, lhs( row ), delta() ) ||
             GT( val, rhs( row ), delta() ) )
         {
            // Minor rhs violations happen frequently, so print these
            // warnings only with verbose level INFO2 and higher.
            MSG_INFO2( spxout << "WSOLVE88 Warning! Constraint " << row
                              << " is violated by solution" << std::endl
                              << "   lhs:" << lhs( row )
                              << " <= val:" << val
                              << " <= rhs:" << rhs( row ) << std::endl; )

            if( type() == LEAVE && isRowBasic( row ) )
            {
               // find basis variable
               for( c = 0; c < nRows(); ++c )
                  if (basis().baseId(c).isSPxRowId()     
                     && (number(basis().baseId(c)) == row))
                     break;

               assert( c < nRows() );

               MSG_WARNING( spxout << "WSOLVE90 basis idx:" << c
                                   << " fVec:" << fVec()[c]
                                   << " fRhs:" << fRhs()[c]
                                   << " fTest:" << fTest()[c] << std::endl; )
            }
         }
      }
      for(int col = 0; col < nCols(); ++col )
      {
         if( LT( sol[col], lower( col ), delta() ) ||
             GT( sol[col], upper( col ), delta() ) )
         {
            // Minor bound violations happen frequently, so print these
            // warnings only with verbose level INFO2 and higher.
            MSG_INFO2( spxout << "WSOLVE91 Warning! Bound for column " << col
                                 << " is violated by solution" << std::endl
                                 << "   lower:" << lower( col )
                                 << " <= val:" << sol[col]
                                 << " <= upper:" << upper( col ) << std::endl; )

            if( type() == LEAVE && isColBasic( col ) )
            {
               for( c = 0; c < nRows() ; ++c)
                  if ( basis().baseId( c ).isSPxColId()    
                     && ( number( basis().baseId( c ) ) == col ))
                     break;

               assert( c < nRows() );
               MSG_WARNING( spxout << "WSOLVE92 basis idx:" << c
                                   << " fVec:" << fVec()[c]
                                   << " fRhs:" << fRhs()[c]
                                   << " fTest:" << fTest()[c] << std::endl; )
            }
         }
      }
   }
#endif  // ENABLE_ADDITIONAL_CHECKS

   return status();
}

void SPxSolver::testVecs()
{
   METHOD( "SPxSolver::testVecs()" );

   assert(SPxBasis::status() > SPxBasis::SINGULAR);

   DVector tmp(dim());

   tmp = *theCoPvec;
   multWithBase(tmp);
   tmp -= *theCoPrhs;
   if (tmp.length() > delta())
   {
      MSG_INFO3( spxout << "ISOLVE93 " << iteration() << ":\tcoP error = \t"
                        << tmp.length() << std::endl; )

      tmp.clear();
      SPxBasis::coSolve(tmp, *theCoPrhs);
      multWithBase(tmp);
      tmp -= *theCoPrhs;
      MSG_INFO3( spxout << "ISOLVE94\t\t" << tmp.length() << std::endl; )

      tmp.clear();
      SPxBasis::coSolve(tmp, *theCoPrhs);
      tmp -= *theCoPvec;
      MSG_INFO3( spxout << "ISOLVE95\t\t" << tmp.length() << std::endl; )
   }

   tmp = *theFvec;
   multBaseWith(tmp);
   tmp -= *theFrhs;
   if (tmp.length() > delta())
   {
      MSG_INFO3( spxout << "ISOLVE96 " << iteration() << ":\t  F error = \t"
                           << tmp.length() << std::endl; )

      tmp.clear();
      SPxBasis::solve(tmp, *theFrhs);
      tmp -= *theFvec;
      MSG_INFO3( spxout << "ISOLVE97\t\t" << tmp.length() << std::endl; )
   }

#ifdef ENABLE_ADDITIONAL_CHECKS
   if (type() == ENTER)
   {
      for (int i = 0; i < dim(); ++i)
      {
         if (theCoTest[i] < -delta() && isCoBasic(i))
         {
            /// @todo Error message "this shalt not be": shalt this be an assert (also below)?
            MSG_ERROR( spxout << "ESOLVE98 testVecs: theCoTest: this shalt not be!"
                              << std::endl
                              << "  i=" << i 
                              << ", theCoTest[i]=" << theCoTest[i]
                              << ", delta()=" << delta() << std::endl; )
         }
      }

      for (int i = 0; i < coDim(); ++i)
      {
         if (theTest[i] < -delta() && isBasic(i))
         {
            MSG_ERROR( spxout << "ESOLVE99 testVecs: theTest: this shalt not be!"
                              << std::endl
                              << "  i=" << i 
                              << ", theTest[i]=" << theTest[i]
                              << ", delta()=" << delta() << std::endl; )
         }
      }
   }
#endif // ENABLE_ADDITIONAL_CHECKS
}

bool SPxSolver::terminate()
{
   METHOD( "SPxSolver::terminate()" );
#ifdef ENABLE_ADDITIONAL_CHECKS
   if (SPxBasis::status() > SPxBasis::SINGULAR)
      testVecs();
#endif

   int redo = dim();

   if (redo < 1000)
      redo = 1000;

   if (iteration() > 10 && iteration() % redo == 0)
   {
#ifdef ENABLE_ADDITIONAL_CHECKS
      DVector cr(*theCoPrhs);
      DVector fr(*theFrhs);
#endif 

      if (type() == ENTER)
         computeEnterCoPrhs();
      else
         computeLeaveCoPrhs();

      computeFrhs();

#ifdef ENABLE_ADDITIONAL_CHECKS
      cr -= *theCoPrhs;
      fr -= *theFrhs;
      if (cr.length() > delta())
         MSG_WARNING( spxout << "WSOLVE50 unexpected change of coPrhs " 
                             << cr.length() << std::endl; )
      if (fr.length() > delta())
         MSG_WARNING( spxout << "WSOLVE51 unexpected change of   Frhs " 
                             << fr.length() << std::endl; )
#endif

      if (updateCount > 1)
      {
         MSG_INFO3( spxout << "ISOLVE52 terminate triggers refactorization" 
                           << std::endl; )
         factorize();
      }
      SPxBasis::coSolve(*theCoPvec, *theCoPrhs);
      SPxBasis::solve (*theFvec, *theFrhs);

      if (pricing() == FULL)
      {
         computePvec();
         if (type() == ENTER)
            computeTest();
      }

      if (shift() > 0.0)
         unShift();
   }

   if ( maxTime >= 0 && maxTime < infinity && time() >= maxTime )
   {
      MSG_INFO2( spxout << "ISOLVE54 Timelimit (" << maxTime
                        << ") reached" << std::endl; )
      m_status = ABORT_TIME;
      return true;   
   }

   // objLimit is set and we are running DUAL:
   // - objLimit is set if objLimit < infinity
   // - DUAL is running if rep() * type() > 0 == DUAL (-1 == PRIMAL)
   //
   // In this case we have given a objective value limit, e.g, through a
   // MIP solver, and we want stop solving the LP if we figure out that the
   // optimal value of the current LP can not be better then this objective
   // limit. More precisely:
   // - MINIMIZATION Problem
   //   We want stop the solving process if
   //   objLimit <= current objective value of the DUAL LP
   // - MAXIMIZATION Problem
   //   We want stop the solving process if 
   //   objLimit >= current objective value of the DUAL LP
   if (objLimit < infinity && type() * rep() > 0)
   {
      // We have no bound shifts; therefore, we can trust the current
      // objective value.
      // It might be even possible to use this termination value in case of
      // bound violations (shifting) but in this case it is quite difficult
      // to determine if we already reached the limit.
      if( shift() < epsilon() && maxInfeas() + shift() <= delta() )
      {
         // SPxSense::MINIMIZE == -1, so we have sign = 1 on minimizing
         if( spxSense() * value() <= spxSense() * objLimit ) 
         {
            MSG_INFO2( spxout << "ISOLVE55 Objective value limit (" << objLimit
               << ") reached" << std::endl; )
            MSG_DEBUG(
               spxout << "DSOLVE56 Objective value limit reached" << std::endl
                      << " (value: " << value()
                      << ", limit: " << objLimit << ")" << std::endl
                      << " (spxSense: " << int(spxSense())
                      << ", rep: " << int(rep())
                      << ", type: " << int(type()) << ")" << std::endl;
            )
            
            m_status = ABORT_VALUE;
            return true;
         }
      }
   }

   if( SPxBasis::status() >= SPxBasis::OPTIMAL  ||
       SPxBasis::status() <= SPxBasis::SINGULAR )
   {
      m_status = UNKNOWN;
      return true;
   }
   return false;
}

SPxSolver::Status SPxSolver::getPrimal (Vector& p_vector) const
{
   METHOD( "SPxSolver::getPrimal()" );

   if (!isInitialized())
   {
      /* exit if presolving/simplifier cleared the problem */
      if (status() == NO_PROBLEM)
         return status();
      throw SPxStatusException("XSOLVE06 Not Initialized");
   }
   if (rep() == ROW)
      p_vector = coPvec();
   else
   {
      const SPxBasis::Desc& ds = desc();

      for (int i = 0; i < nCols(); ++i)
      {
         switch (ds.colStatus(i))
         {
         case SPxBasis::Desc::P_ON_LOWER :
            p_vector[i] = SPxLP::lower(i);
            break;
         case SPxBasis::Desc::P_ON_UPPER :
         case SPxBasis::Desc::P_FIXED :
            p_vector[i] = SPxLP::upper(i);
            break;
         case SPxBasis::Desc::P_FREE :
            p_vector[i] = 0;
            break;
         case SPxBasis::Desc::D_FREE :
         case SPxBasis::Desc::D_ON_UPPER :
         case SPxBasis::Desc::D_ON_LOWER :
         case SPxBasis::Desc::D_ON_BOTH :
         case SPxBasis::Desc::D_UNDEFINED :
            break;
         default:
            throw SPxInternalCodeException("XSOLVE07 This should never happen.");
         }
      }
      for (int j = 0; j < dim(); ++j)
      {
         if (baseId(j).isSPxColId())
            p_vector[ number(SPxColId(baseId(j))) ] = fVec()[j];
      }
   }
   return status();
}

SPxSolver::Status SPxSolver::getDual (Vector& p_vector) const
{
   METHOD( "SPxSolver::getDual()" );

   assert(isInitialized());

   if (!isInitialized()) 
   {
      /* exit if presolving/simplifier cleared the problem */
      if (status() == NO_PROBLEM)
         return status();
      throw SPxStatusException("XSOLVE08 No Problem loaded");
   }

   if (rep() == ROW)
   {
      int i;
      p_vector.clear ();
      for (i = nCols() - 1; i >= 0; --i)
      {
         if (baseId(i).isSPxRowId())
            p_vector[ number(SPxRowId(baseId(i))) ] = fVec()[i];
      }
   }
   else
      p_vector = coPvec();

   p_vector *= Real(spxSense());

   return status();
}

SPxSolver::Status SPxSolver::getRedCost (Vector& p_vector) const
{
   METHOD( "SPxSolver::getRedCost()" );

   assert(isInitialized());

   if (!isInitialized())
   {
      throw SPxStatusException("XSOLVE09 No Problem loaded");    
      // return NOT_INIT;
   }

   if (rep() == ROW)
   {
      int i;
      p_vector.clear();
      if (spxSense() == SPxLP::MINIMIZE)
      {
         for (i = dim() - 1; i >= 0; --i)
         {
            if (baseId(i).isSPxColId())
               p_vector[ number(SPxColId(baseId(i))) ] = -fVec()[i];
         }
      }
      else
      {
         for (i = dim() - 1; i >= 0; --i)
         {
            if (baseId(i).isSPxColId())
               p_vector[ number(SPxColId(baseId(i))) ] = fVec()[i];
         }
      }
   }
   else
   {
      p_vector = maxObj();
      p_vector -= pVec();
      if (spxSense() == SPxLP::MINIMIZE)
         p_vector *= -1.0;
   }

   return status();
}

SPxSolver::Status SPxSolver::getPrimalray (Vector& p_vector) const
{
   METHOD( "SPxSolver::getPrimalray()" );

   assert(isInitialized());

   if (!isInitialized())
   {
      throw SPxStatusException("XSOLVE10 No Problem loaded");
      // return NOT_INIT;
   }

   assert(SPxBasis::status() == SPxBasis::UNBOUNDED);
   p_vector.clear();
   p_vector = primalRay;

   return status();
}

SPxSolver::Status SPxSolver::getDualfarkas (Vector& p_vector) const
{
   METHOD( "SPxSolver::getDualfarkas()" );

   assert(isInitialized());

   if (!isInitialized())
   {
      throw SPxStatusException("XSOLVE10 No Problem loaded");
      // return NOT_INIT;
   }

   assert(SPxBasis::status() == SPxBasis::INFEASIBLE);
   p_vector.clear();
   p_vector = dualFarkas;

   return status();
}

SPxSolver::Status SPxSolver::getSlacks (Vector& p_vector) const
{
   METHOD( "SPxSolver::getSlacks()" );

   assert(isInitialized());

   if (!isInitialized())
   {
      throw SPxStatusException("XSOLVE11 No Problem loaded");
      // return NOT_INIT;
   }

   if (rep() == COLUMN)
   {
      int i;
      const SPxBasis::Desc& ds = desc();
      for (i = nRows() - 1; i >= 0; --i)
      {
         switch (ds.rowStatus(i))
         {
         case SPxBasis::Desc::P_ON_LOWER :
            p_vector[i] = lhs(i);
            break;
         case SPxBasis::Desc::P_ON_UPPER :
         case SPxBasis::Desc::P_FIXED :
            p_vector[i] = rhs(i);
            break;
         case SPxBasis::Desc::P_FREE :
            p_vector[i] = 0;
            break;
         case SPxBasis::Desc::D_FREE :
         case SPxBasis::Desc::D_ON_UPPER :
         case SPxBasis::Desc::D_ON_LOWER :
         case SPxBasis::Desc::D_ON_BOTH :
         case SPxBasis::Desc::D_UNDEFINED :
            break;
         default:
            throw SPxInternalCodeException("XSOLVE12 This should never happen.");
         }
      }
      for (i = dim() - 1; i >= 0; --i)
      {
         if (baseId(i).isSPxRowId())
            p_vector[ number(SPxRowId(baseId(i))) ] = -(*theFvec)[i];
      }
   }
   else
      p_vector = pVec();

   return status();
}

SPxSolver::Status SPxSolver::status() const
{
   METHOD( "SPxSolver::status()" );
   switch( m_status )
   {
   case UNKNOWN :      
      switch (SPxBasis::status())
      {
      case SPxBasis::NO_PROBLEM :
         return NO_PROBLEM;
      case SPxBasis::SINGULAR :
         return SINGULAR;
      case SPxBasis::REGULAR :
      case SPxBasis::DUAL :
      case SPxBasis::PRIMAL :
         return UNKNOWN;
      case SPxBasis::OPTIMAL :
         return OPTIMAL;
      case SPxBasis::UNBOUNDED :
         return UNBOUNDED;
      case SPxBasis::INFEASIBLE :
         return INFEASIBLE;
      default:
         return ERROR;
      }
   case SINGULAR : 
      return m_status;
   case OPTIMAL :
      assert( SPxBasis::status() == SPxBasis::OPTIMAL );
      /*lint -fallthrough*/
   case ABORT_CYCLING :
   case ABORT_TIME :
   case ABORT_ITER :
   case ABORT_VALUE :
   case RUNNING :
   case REGULAR :
   case NOT_INIT :
   case NO_SOLVER :
   case NO_PRICER :
   case NO_RATIOTESTER :
   case ERROR:
      return m_status;
   default:
      return ERROR;
   }
}

SPxSolver::Status SPxSolver::getResult(
   Real* p_value,
   Vector* p_primal,
   Vector* p_slacks,
   Vector* p_dual,
   Vector* reduCosts) const
{
   METHOD( "SPxSolver::getResult()" );
   if (p_value)
      *p_value = this->value();
   if (p_primal)
      getPrimal(*p_primal);
   if (p_slacks)
      getSlacks(*p_slacks);
   if (p_dual)
      getDual(*p_dual);
   if (reduCosts)
      getRedCost(*reduCosts);
   return status();
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
