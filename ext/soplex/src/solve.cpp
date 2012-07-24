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

#include "spxdefines.h"
#include "clufactor.h"

namespace soplex
{

void CLUFactor::solveUright(Real* wrk, Real* vec) const
{
   METHOD( "CLUFactor::solveUright()" );

   for(int i = thedim - 1; i >= 0; i--)
   {
      int  r = row.orig[i];
      int  c = col.orig[i];
      Real x = wrk[c] = diag[r] * vec[r];

      vec[r] = 0.0;

      if (x != 0.0)
         //if (isNotZero(x))
      {
         for(int j = u.col.start[c]; j < u.col.start[c] + u.col.len[c]; j++)
            vec[u.col.idx[j]] -= x * u.col.val[j];
      }
   }
}

int CLUFactor::solveUrightEps(Real* vec, int* nonz, Real eps, Real* rhs)
{
   METHOD( "CLUFactor::solveUrightEps()" );
   int i, j, r, c, n;
   int *rorig, *corig;
   int *cidx, *clen, *cbeg;
   Real *cval;
   Real x;

   int *idx;
   Real *val;

   rorig = row.orig;
   corig = col.orig;

   cidx = u.col.idx;
   cval = u.col.val;
   clen = u.col.len;
   cbeg = u.col.start;

   n = 0;

   for (i = thedim - 1; i >= 0; --i)
   {
      r = rorig[i];
      x = diag[r] * rhs[r];
      if (isNotZero(x, eps))
      {
         c = corig[i];
         vec[c] = x;
         nonz[n++] = c;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];
         while (j-- > 0)
            rhs[*idx++] -= x * (*val++);
      }
   }

   return n;
}

void CLUFactor::solveUright2(
   Real* p_work1, Real* vec1, Real* p_work2, Real* vec2)
{
   METHOD( "CLUFactor::solveUright2()" );
   int i, j, r, c;
   int *rorig, *corig;
   int *cidx, *clen, *cbeg;
   Real *cval;
   Real x1, x2;

   int* idx;
   Real* val;

   rorig = row.orig;
   corig = col.orig;

   cidx = u.col.idx;
   cval = u.col.val;
   clen = u.col.len;
   cbeg = u.col.start;

   for (i = thedim - 1; i >= 0; --i)
   {
      r = rorig[i];
      c = corig[i];
      p_work1[c] = x1 = diag[r] * vec1[r];
      p_work2[c] = x2 = diag[r] * vec2[r];
      vec1[r] = vec2[r] = 0;
      if (x1 != 0.0 && x2 != 0.0)
      {
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];
         while (j-- > 0)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx++] -= x2 * (*val++);
         }
      }
      else if (x1 != 0.0)
      {
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];
         while (j-- > 0)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if (x2 != 0.0)
      {
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];
         while (j-- > 0)
            vec2[*idx++] -= x2 * (*val++);
      }
   }
}

int CLUFactor::solveUright2eps(
   Real* p_work1, Real* vec1, Real* p_work2, Real* vec2,
   int* nonz, Real eps)
{
   METHOD( "CLUFactor::solveUright2eps()" );
   int i, j, r, c, n;
   int *rorig, *corig;
   int *cidx, *clen, *cbeg;
   bool notzero1, notzero2;
   Real *cval;
   Real x1, x2;

   int* idx;
   Real* val;

   rorig = row.orig;
   corig = col.orig;

   cidx = u.col.idx;
   cval = u.col.val;
   clen = u.col.len;
   cbeg = u.col.start;

   n = 0;

   for (i = thedim - 1; i >= 0; --i)
   {
      c = corig[i];
      r = rorig[i];
      p_work1[c] = x1 = diag[r] * vec1[r];
      p_work2[c] = x2 = diag[r] * vec2[r];
      vec1[r] = vec2[r] = 0;
      notzero1 = isNotZero(x1, eps);
      notzero2 = isNotZero(x2, eps);

      if (notzero1 && notzero2)
      {
         *nonz++ = c;
         n++;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];
         while (j-- > 0)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx++] -= x2 * (*val++);
         }
      }
      else if (notzero1)
      {
         p_work2[c] = 0;
         *nonz++ = c;
         n++;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];
         while (j-- > 0)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if (notzero2)
      {
         p_work1[c] = 0;
         val = &cval[cbeg[c]];
         idx = &cidx[cbeg[c]];
         j = clen[c];
         while (j-- > 0)
            vec2[*idx++] -= x2 * (*val++);
      }
      else
      {
         p_work1[c] = 0;
         p_work2[c] = 0;
      }
   }
   return n;
}

void CLUFactor::solveLright(Real* vec)
{
   METHOD( "CLUFactor::solveLright()" );
   int i, j, k;
   int end;
   Real x;
   Real *lval, *val;
   int *lrow, *lidx, *idx;
   int *lbeg;

   lval = l.val;
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;
   for (i = 0; i < end; ++i)
   {
      if ((x = vec[lrow[i]]) != 0.0)
      {
         MSG_DEBUG( spxout << "y" << lrow[i] << "=" << vec[lrow[i]] << std::endl; )

         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
         {
            MSG_DEBUG( spxout << "                         -> y" << *idx << " -= " << x << " * " << *val << " = " << x * (*val) << "    -> " << vec[*idx] - x * (*val) << std::endl; )
            vec[*idx++] -= x * (*val++);
         }
      }
   }

   if (l.updateType)                     /* Forest-Tomlin Updates */
   {
      MSG_DEBUG( spxout << "performing FT updates..." << std::endl; )

      end = l.firstUnused;
      for (; i < end; ++i)
      {
         x = 0;
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
            x += vec[*idx++] * (*val++);
         vec[lrow[i]] -= x;

         MSG_DEBUG( spxout << "y" << lrow[i] << "=" << vec[lrow[i]] << std::endl; )
      }

      MSG_DEBUG( spxout << "finished FT updates." << std::endl; )
   }
}

void CLUFactor::solveLright2(Real* vec1, Real* vec2)
{
   METHOD( "CLUFactor::solveLright2()" );
   int i, j, k;
   int end;
   Real x2;
   Real x1;
   Real *lval, *val;
   int *lrow, *lidx, *idx;
   int *lbeg;

   lval = l.val;
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;
   for (i = 0; i < end; ++i)
   {
      x1 = vec1[lrow[i]];
      x2 = vec2[lrow[i]];
      if (x1 != 0 && x2 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx++] -= x2 * (*val++);
         }
      }
      else if (x1 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if (x2 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
            vec2[*idx++] -= x2 * (*val++);
      }
   }

   if (l.updateType)                     /* Forest-Tomlin Updates */
   {
      end = l.firstUnused;
      for (; i < end; ++i)
      {
         x1 = 0;
         x2 = 0;
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
         {
            x1 += vec1[*idx] * (*val);
            x2 += vec2[*idx++] * (*val++);
         }
         vec1[lrow[i]] -= x1;
         vec2[lrow[i]] -= x2;
      }
   }
}

void CLUFactor::solveUpdateRight(Real* vec)
{
   METHOD( "CLUFactor::solveUpdateRight()" );
   int i, j, k;
   int end;
   Real x;
   Real *lval, *val;
   int *lrow, *lidx, *idx;
   int *lbeg;

   assert(!l.updateType);               /* no Forest-Tomlin Updates */

   lval = l.val;
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUnused;
   for (i = l.firstUpdate; i < end; ++i)
   {
      if ((x = vec[lrow[i]]) != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
            vec[*idx++] -= x * (*val++);
      }
   }
}

void CLUFactor::solveUpdateRight2(Real* vec1, Real* vec2)
{
   METHOD( "CLUFactor::solveUpdateRight2()" );
   int i, j, k;
   int end;
   Real x1, x2;
   Real *lval;
   int *lrow, *lidx;
   int *lbeg;

   int* idx;
   Real* val;

   assert(!l.updateType);               /* no Forest-Tomlin Updates */

   lval = l.val;
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUnused;
   for (i = l.firstUpdate; i < end; ++i)
   {
      x1 = vec1[lrow[i]];
      x2 = vec2[lrow[i]];
      if (x1 != 0.0 && x2 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx++] -= x2 * (*val++);
         }
      }
      else if (x1 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if (x2 != 0.0)
      {
         k = lbeg[i];
         idx = &(lidx[k]);
         val = &(lval[k]);
         for (j = lbeg[i + 1]; j > k; --j)
            vec2[*idx++] -= x2 * (*val++);
      }
   }
}

int CLUFactor::solveRight4update(Real* vec, int* nonz, Real eps, 
   Real* rhs, Real* forest, int* forestNum, int* forestIdx)
{
   METHOD( "CLUFactor::solveRight4update()" );
   solveLright(rhs);

   if (forest)
   {
      int n = 0;

      for(int i = 0; i < thedim; i++)
      {
         forestIdx[n] = i;
         forest[i]    = rhs[i];
         n           += rhs[i] != 0.0 ? 1 : 0;
      }
      *forestNum = n;
   }

   if (!l.updateType)            /* no Forest-Tomlin Updates */
   {
      solveUright(vec, rhs);
      solveUpdateRight(vec);
      return 0;
   }
   else
      return solveUrightEps(vec, nonz, eps, rhs);
}

void CLUFactor::solveRight(Real* vec, Real* rhs)
{
   METHOD( "CLUFactor::solveRight()" );
   solveLright(rhs);
   solveUright(vec, rhs);
   if (!l.updateType)            /* no Forest-Tomlin Updates */
      solveUpdateRight(vec);
}

int CLUFactor::solveRight2update(Real* vec1,
                       Real* vec2,
                       Real* rhs1,
                       Real* rhs2,
                       int* nonz,
                       Real eps,
                       Real* forest,
                       int* forestNum,
                       int* forestIdx)
{
   METHOD( "CLUFactor::solveRight2update()" );
   solveLright2(rhs1, rhs2);

   if (forest)
   {
      int n = 0;
      
      for(int i = 0; i < thedim; i++)
      {
         forestIdx[n] = i;
         forest[i]    = rhs1[i];
         n           += rhs1[i] != 0.0 ? 1 : 0;
      }
      *forestNum = n;
   }

   if (!l.updateType)           /* no Forest-Tomlin Updates */
   {
      solveUright2(vec1, rhs1, vec2, rhs2);
      solveUpdateRight2(vec1, vec2);
      return 0;
   }
   else
      return solveUright2eps(vec1, rhs1, vec2, rhs2, nonz, eps);
}

void CLUFactor::solveRight2(
   Real* vec1,
   Real* vec2,
   Real* rhs1,
   Real* rhs2)
{
   METHOD( "CLUFactor::solveRight2()" );
   solveLright2(rhs1, rhs2);
   if (l.updateType)             /* Forest-Tomlin Updates */
      solveUright2(vec1, rhs1, vec2, rhs2);
   else
   {
      solveUright2(vec1, rhs1, vec2, rhs2);
      solveUpdateRight2(vec1, vec2);
   }
}

/*****************************************************************************/
#if 0
void CLUFactor::solveUleft(Real* p_work, Real* vec)
{
   METHOD( "CLUFactor::solveUleft()" );
   Real x;
   int i, k, r, c;
   int *rorig, *corig;
   int *ridx, *rlen, *rbeg, *idx;
   Real *rval, *val;

   rorig = row.orig;
   corig = col.orig;

   ridx = u.row.idx;
   rval = u.row.val;
   rlen = u.row.len;
   rbeg = u.row.start;

   for (i = 0; i < thedim; ++i)
   {
      c = corig[i];
      r = rorig[i];
      x = vec[c];
      vec[c] = 0;
      if (x != 0.0)
      {
         x *= diag[r];
         p_work[r] = x;
         k = rbeg[r];
         idx = &ridx[k];
         val = &rval[k];
         for (int m = rlen[r]; m != 0; --m)
            vec[*idx++] -= x * (*val++);
      }
   }
}
#else
void CLUFactor::solveUleft(Real* p_work, Real* vec)
{
   METHOD( "CLUFactor::solveUleft()" );
#if 0
   for (int i = 0; i < thedim; ++i)
   {
      int  r  = row.orig[i];
      int end = u.row.start[r] + u.row.len[r];

      for(int m = u.row.start[r]; m < end; m++)
      {
         std::cout << u.row.val[m] << " ";
      }
      std::cout << "\n";
   }
#endif
   for (int i = 0; i < thedim; ++i)
   {
      int  c  = col.orig[i];
      int  r  = row.orig[i];

      assert(c >= 0);  // Inna/Tobi: otherwise, vec[c] would be strange...
      assert(r >= 0);  // Inna/Tobi: otherwise, diag[r] would be strange...

      Real x  = vec[c];

      ASSERT_WARN( "WSOLVE01", fabs(x) < 1e40 );
      ASSERT_WARN( "WSOLVE02", fabs(vec[c]) < 1e40 );

#if defined(WITH_WARNINGS) || !defined(NDEBUG)
      Real y = vec[c];
#endif

      vec[c]  = 0.0;

      if (x != 0.0)
      {
         ASSERT_WARN( "WSOLVE03", fabs(diag[r]) < 1e40 );

         x        *= diag[r];
         p_work[r] = x;

         ASSERT_WARN( "WSOLVE04", fabs(x) < 1e40 );

         int end = u.row.start[r] + u.row.len[r];

         for(int m = u.row.start[r]; m < end; m++)
         {
            ASSERT_WARN( "WSOLVE05", fabs(u.row.val[m]) < 1e40 );
            ASSERT_WARN( "WSOLVE06", fabs(vec[u.row.idx[m]]) < infinity );
            vec[u.row.idx[m]] -= x * u.row.val[m];
            ASSERT_WARN( "WSOLVE07", fabs(vec[u.row.idx[m]]) < 1e40 );
         }
      }
      ASSERT_WARN( "WSOLVE08", fabs(y) < 1e40 );
   }
}
#endif

void CLUFactor::solveUleft2(
   Real* p_work1, Real* vec1, Real* p_work2, Real* vec2)
{
   METHOD( "CLUFactor::solveUleft2()" );
   Real x1;
   Real x2;
   int i, k, r, c;
   int *rorig, *corig;
   int *ridx, *rlen, *rbeg, *idx;
   Real *rval, *val;

   rorig = row.orig;
   corig = col.orig;

   ridx = u.row.idx;
   rval = u.row.val;
   rlen = u.row.len;
   rbeg = u.row.start;

   for (i = 0; i < thedim; ++i)
   {
      c = corig[i];
      r = rorig[i];
      x1 = vec1[c];
      x2 = vec2[c];
      if ((x1 != 0.0) && (x2 != 0.0))
      {
         x1 *= diag[r];
         x2 *= diag[r];
         p_work1[r] = x1;
         p_work2[r] = x2;
         k = rbeg[r];
         idx = &ridx[k];
         val = &rval[k];
         for (int m = rlen[r]; m != 0; --m)
         {
            vec1[*idx] -= x1 * (*val);
            vec2[*idx] -= x2 * (*val++);
            idx++;
         }
      }
      else if (x1 != 0.0)
      {
         x1 *= diag[r];
         p_work1[r] = x1;
         k = rbeg[r];
         idx = &ridx[k];
         val = &rval[k];
         for (int m = rlen[r]; m != 0; --m)
            vec1[*idx++] -= x1 * (*val++);
      }
      else if (x2 != 0.0)
      {
         x2 *= diag[r];
         p_work2[r] = x2;
         k = rbeg[r];
         idx = &ridx[k];
         val = &rval[k];
         for (int m = rlen[r]; m != 0; --m)
            vec2[*idx++] -= x2 * (*val++);
      }
   }
}

int CLUFactor::solveLleft2forest(
   Real* vec1,
   int* /* nonz */,
   Real* vec2,
   Real /* eps */)
{
   METHOD( "CLUFactor::solveLleft2forest()" );
   int i;
   int j;
   int k;
   int end;
   Real x1, x2;
   Real *lval, *val;
   int *lidx, *idx, *lrow;
   int *lbeg;

   lval = l.val;
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;
   for (i = l.firstUnused - 1; i >= end; --i)
   {
      j = lrow[i];
      x1 = vec1[j];
      x2 = vec2[j];
      if (x1 != 0.0)
      {
         if (x2 != 0.0)
         {
            k = lbeg[i];
            val = &lval[k];
            idx = &lidx[k];
            for (j = lbeg[i + 1]; j > k; --j)
            {
               vec1[*idx] -= x1 * (*val);
               vec2[*idx++] -= x2 * (*val++);
            }
         }
         else
         {
            k = lbeg[i];
            val = &lval[k];
            idx = &lidx[k];
            for (j = lbeg[i + 1]; j > k; --j)
               vec1[*idx++] -= x1 * (*val++);
         }
      }
      else if (x2 != 0.0)
      {
         k = lbeg[i];
         val = &lval[k];
         idx = &lidx[k];
         for (j = lbeg[i + 1]; j > k; --j)
            vec2[*idx++] -= x2 * (*val++);
      }
   }
   return 0;
}

void CLUFactor::solveLleft2(
   Real* vec1,
   int* /* nonz */,
   Real* vec2,
   Real /* eps */)
{
   METHOD( "CLUFactor::solveLleft2()" );
   int i, j, k, r;
   int x1not0, x2not0;
   Real x1, x2;

   Real *rval, *val;
   int *ridx, *idx;
   int *rbeg;
   int *rorig;

   ridx  = l.ridx;
   rval  = l.rval;
   rbeg  = l.rbeg;
   rorig = l.rorig;

#ifndef WITH_L_ROWS
   Real*   lval  = l.val;
   int*    lidx  = l.idx;
   int*    lrow  = l.row;
   int*    lbeg  = l.start;

   i = l.firstUpdate - 1;
   for (; i >= 0; --i)
   {
      k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x1 = 0;
      x2 = 0;
      for (j = lbeg[i + 1]; j > k; --j)
      {
         x1 += vec1[*idx] * (*val);
         x2 += vec2[*idx++] * (*val++);
      }
      vec1[lrow[i]] -= x1;
      vec2[lrow[i]] -= x2;
   }
#else
   for (i = thedim; i--;)
   {
      r = rorig[i];
      x1 = vec1[r];
      x2 = vec2[r];
      x1not0 = (x1 != 0);
      x2not0 = (x2 != 0);

      if (x1not0 && x2not0)
      {
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];
         while (j-- > 0)
         {
            assert(row.perm[*idx] < i);
            vec1[*idx] -= x1 * *val;
            vec2[*idx++] -= x2 * *val++;
         }
      }
      else if (x1not0)
      {
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];
         while (j-- > 0)
         {
            assert(row.perm[*idx] < i);
            vec1[*idx++] -= x1 * *val++;
         }
      }
      else if (x2not0)
      {
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];
         while (j-- > 0)
         {
            assert(row.perm[*idx] < i);
            vec2[*idx++] -= x2 * *val++;
         }
      }
   }
#endif
}

int CLUFactor::solveLleftForest(Real* vec, int* /* nonz */, Real /* eps */)
{
   METHOD( "CLUFactor::solveLleftForest()" );
   int i, j, k, end;
   Real x;
   Real *val, *lval;
   int *idx, *lidx, *lrow, *lbeg;

   lval = l.val;
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   end = l.firstUpdate;
   for (i = l.firstUnused - 1; i >= end; --i)
   {
      if ((x = vec[lrow[i]]) != 0.0)
      {
         k = lbeg[i];
         val = &lval[k];
         idx = &lidx[k];
         for (j = lbeg[i + 1]; j > k; --j)
            vec[*idx++] -= x * (*val++);
      }
   }

   return 0;
}

void CLUFactor::solveLleft(Real* vec) const
{
   METHOD( "CLUFactor::solveLleft()" );

#ifndef WITH_L_ROWS
   int*  idx;
   Real* val;
   Real* lval  = l.val;
   int*  lidx  = l.idx;
   int*  lrow  = l.row;
   int*  lbeg  = l.start;
   
   for(int i = l.firstUpdate - 1; i >= 0; --i)
   {
      int k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x = 0;
      for(int j = lbeg[i + 1]; j > k; --j)
         x += vec[*idx++] * (*val++);
      vec[lrow[i]] -= x;
   }
#else
   for(int i = thedim - 1; i >= 0; --i)
   {
      int  r = l.rorig[i];
      Real x = vec[r];

      if (x != 0.0)
      {
         for(int k = l.rbeg[r]; k < l.rbeg[r + 1]; k++)
         { 
            int j = l.ridx[k];

            assert(l.rperm[j] < i);

            vec[j] -= x * l.rval[k];
         }
      }
   }
#endif // WITH_L_ROWS
}

int CLUFactor::solveLleftEps(Real* vec, int* nonz, Real eps)
{
   METHOD( "CLUFactor::solveLleftEps()" );
   int i, j, k, n;
   int r;
   Real x;
   Real *rval, *val;
   int *ridx, *idx;
   int *rbeg;
   int* rorig;

   ridx = l.ridx;
   rval = l.rval;
   rbeg = l.rbeg;
   rorig = l.rorig;
   n = 0;
#ifndef WITH_L_ROWS
   Real* lval = l.val;
   int*  lidx = l.idx;
   int*  lrow = l.row;
   int*  lbeg = l.start;

   for (i = l.firstUpdate - 1; i >= 0; --i)
   {
      k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x = 0;
      for (j = lbeg[i + 1]; j > k; --j)
         x += vec[*idx++] * (*val++);
      vec[lrow[i]] -= x;
   }
#else
   for (i = thedim; i--;)
   {
      r = rorig[i];
      x = vec[r];
      if (isNotZero(x, eps))
      {
         *nonz++ = r;
         n++;
         k = rbeg[r];
         j = rbeg[r + 1] - k;
         val = &rval[k];
         idx = &ridx[k];
         while (j-- > 0)
         {
            assert(row.perm[*idx] < i);
            vec[*idx++] -= x * *val++;
         }
      }
      else
         vec[r] = 0;
   }
#endif

   return n;
}

void CLUFactor::solveUpdateLeft(Real* vec)
{
   METHOD( "CLUFactor::solveUpdateLeft()" );
   int i, j, k, end;
   Real x;
   Real *lval, *val;
   int *lrow, *lidx, *idx;
   int *lbeg;

   lval = l.val;
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   assert(!l.updateType);               /* Forest-Tomlin Updates */

   end = l.firstUpdate;
   for (i = l.firstUnused - 1; i >= end; --i)
   {
      k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x = 0;
      for (j = lbeg[i + 1]; j > k; --j)
         x += vec[*idx++] * (*val++);
      vec[lrow[i]] -= x;
   }
}

void CLUFactor::solveUpdateLeft2(Real* vec1, Real* vec2)
{
   METHOD( "CLUFactor::solveUpdateLeft2()" );
   int i, j, k, end;
   Real x1, x2;
   Real *lval, *val;
   int *lrow, *lidx, *idx;
   int *lbeg;

   lval = l.val;
   lidx = l.idx;
   lrow = l.row;
   lbeg = l.start;

   assert(!l.updateType);               /* Forest-Tomlin Updates */

   end = l.firstUpdate;
   for (i = l.firstUnused - 1; i >= end; --i)
   {
      k = lbeg[i];
      val = &lval[k];
      idx = &lidx[k];
      x1 = 0;
      x2 = 0;
      for (j = lbeg[i + 1]; j > k; --j)
      {
         x1 += vec1[*idx] * (*val);
         x2 += vec2[*idx++] * (*val++);
      }
      vec1[lrow[i]] -= x1;
      vec2[lrow[i]] -= x2;
   }
}

void CLUFactor::solveLeft(Real* vec, Real* rhs)
{
   METHOD( "CLUFactor::solveLeft()" );

   if (!l.updateType)            /* no Forest-Tomlin Updates */
   {
      solveUpdateLeft(rhs);
      solveUleft(vec, rhs);
      solveLleft(vec);
   }
   else
   {
      solveUleft(vec, rhs);
      //@todo is 0.0 the right value for eps here ?
      solveLleftForest(vec, 0, 0.0);
      solveLleft(vec);
   }
}

int CLUFactor::solveLeftEps(Real* vec, Real* rhs, int* nonz, Real eps)
{
   METHOD( "CLUFactor::solveLeftEps()" );
   if (!l.updateType)            /* no Forest-Tomlin Updates */
   {
      solveUpdateLeft(rhs);
      solveUleft(vec, rhs);
      return solveLleftEps(vec, nonz, eps);
   }
   else
   {
      solveUleft(vec, rhs);
      solveLleftForest(vec, nonz, eps);
      return solveLleftEps(vec, nonz, eps);
   }
}

int CLUFactor::solveLeft2(
   Real* vec1,
   int* nonz,
   Real* vec2,
   Real eps,
   Real* rhs1,
   Real* rhs2)
{
   METHOD( "CLUFactor::solveLeft2()" );
   if (!l.updateType)            /* no Forest-Tomlin Updates */
   {
      solveUpdateLeft2(rhs1, rhs2);
      solveUleft2(vec1, rhs1, vec2, rhs2);
      solveLleft2(vec1, nonz, vec2, eps);
      return 0;
   }
   else
   {
      solveUleft2(vec1, rhs1, vec2, rhs2);
      solveLleft2forest(vec1, nonz, vec2, eps);
      solveLleft2(vec1, nonz, vec2, eps);
      return 0;
   }
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
