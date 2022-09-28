/* Includes implementation of linear programming using Simplex. */

// TODO: using procedural programming makes sense for competitive programming, but
// given how numkit has changes, convert this to OO, and give standar interface to
// all the algorithms.

#define MAXN 100  // max # of variables
#define MAXC 3000 // max # of constraints
#define EPS 1e-8

// This is the A, b, and c matrix you should populate before calling simplex().
// If an optimum solution was found, it is stored in x[] and has value "val".
double A[MAXC + 1][MAXN + 1], b[MAXC], c[MAXN], x[MAXN], val;

// For internal use only.
int xB[MAXC], xN[MAXN];

/* Variable xB[r] exits the basis, xN[c] enters the basis.
  Comment: The fabs() check is not necessary for correctness (though, the i != r check is).
           However, it speeds it up a lot in sparse instances. For "roadtimes" removing
           that check increases the time from 0.41s to 2.51s.
*/
void pivot(int n, int m, int r, int c)
{
    for (int j = 0; j <= n; ++j)
        if (j != c)
            A[r][j] /= A[r][c];

    for (int i = 0; i <= m; ++i)
        if (i != r && fabs(A[i][c]) > EPS)
            for (int j = 0; j <= n; ++j)
                if (j != c)
                    A[i][j] -= A[r][j] * A[i][c];

    for (int i = 0; i <= m; ++i)
        if (i != r)
            A[i][c] = -A[i][c] / A[r][c];

    A[r][c] = 1 / A[r][c];

    swap(xB[r], xN[c]);
}

// Bland's rule: if two indices are valid choices for the pivot, choose the
//               one with the lexicographically smallest variable
void bland(int a, int &b, int *v)
{
    if (b < 0 || v[a] < v[b])
        b = a;
}

/* Returns true if a feasible basis is found, false if LP is infeasible.
   Idea: Try the all-0 solution (i.e. basis where the x >= 0 constraints are tight).
         Repeatedly pivot out any row with a negative b-entry.
   I don't have a source in mind that proves this works, but come ask me if you want :)
*/
bool phase1(int n, int m)
{
    // pivot while some b-value is negative
    while (true)
    {
        int r = -1, c = -1;

        for (int i = 0; i < m; ++i)
            if (A[i][n] < -EPS)
                bland(i, r, xB);

        if (r == -1)
            return true;

        for (int j = 0; j < n; ++j)
            if (A[r][j] < -EPS)
                bland(j, c, xN);

        if (c == -1)
            return false;

        pivot(n, m, r, c);
    }
}

/*
  Assumes A[i][n] >= 0 for all 0 <= i < m (i.e. current basis is feasible).
  Returns true if an optimum is found, false if the LP is unbounded.
  Currently uses Bland's rule. Should probably test other pivoting procedures.
*/
bool phase2(int n, int m)
{
    // pivot until no improvement
    while (true)
    {
        int r = -1, c = -1;

        // find a column with negative c-coefficient
        // yes, it should be A[m][j] > EPS even though we said "negative",
        // this is because row A[m] is storing -c
        for (int j = 0; j < n; ++j)
            if (A[m][j] > EPS)
                bland(j, c, xN);

        if (c == -1)
            return true;

        for (int i = 0; i < m; ++i)
            if (A[i][c] > EPS)
            {
                if (r == -1)
                    r = i;
                else
                {
                    double lhs = A[i][c] * A[r][n], rhs = A[r][c] * A[i][n];
                    if (lhs > rhs + EPS)
                        r = i;
                    else if (fabs(lhs - rhs) < EPS)
                        bland(i, r, xB);
                }
            }

        if (r == -1)
            return false;

        pivot(n, m, r, c);
    }
}

/* Returns: 1 - opt found (x is this opt and val is its value)
            0 - infeasible, -1 - unbounded
   Two-phase simplex method.
*/
int simplex(int n, int m)
{
    // add b to the end of A and c to the bottom of A
    for (int j = 0; j < n; ++j)
    {
        A[m][j] = c[j];
        xN[j] = j;
    }

    // prepare initial (possibly infeasible) basis
    for (int i = 0; i < m; ++i)
    {
        A[i][n] = b[i];
        xB[i] = i + n;
    }

    A[m][n] = 0; // this will be -(value of x)

    // now pivot to a feasible basis
    if (!phase1(n, m))
        return 0;

    // and then pivot to an optimum basis
    if (!phase2(n, m))
        return -1;

    // recover x and val
    val = -A[m][n];
    for (int i = 0; i < n; ++i)
        x[i] = 0;
    for (int i = 0; i < m; ++i)
        if (xB[i] < n)
            x[xB[i]] = A[i][n];

    return 1;
}
