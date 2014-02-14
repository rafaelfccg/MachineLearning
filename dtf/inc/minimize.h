#include <iostream>
#include <iomanip>
#include <numeric>
#include <cassert>
#include "vec.h"

namespace ts
{
namespace minimize
{
namespace _LBFGS
{
template <typename TScalar, typename TVector, typename Func>
class SimpleLineSearch {
public:
   SimpleLineSearch(Func& prob, const TVector& x0, const TVector& x0_grad,
	             const TVector& H_grad, TScalar x0_fval)
   : prob(prob), x0(x0), x0_grad(x0_grad), H_grad(H_grad),
		   x0_fval(x0_fval), x0_phi_grad(0),
		   evaluation_count(0),
		   xalpha_val(std::numeric_limits<TScalar>::signaling_NaN())
   {
      assert(x0.size() > 0);
      xalpha.resize(x0.size());
      xalphagrad.resize(x0.size());

      // phi'(0) = - p' \nabla_x f(x_k)
      x0_phi_grad = -dot(x0_grad, H_grad);
      assert(x0_phi_grad < 0.0);
   }

   // The initial stepsize tried must be given in alpha.
   // Returns the number of evaluations and alpha in its argument.
   unsigned int ComputeStepLength(TScalar& alpha)
   {
      // Previous alpha
      TScalar alpha_min = static_cast<TScalar>(1.0e-12);
      TScalar alpha_max = static_cast<TScalar>(1e6);

      TScalar fa_enlarge = static_cast<TScalar>(1.7);
      TScalar fa_shrink = static_cast<TScalar>(0.5);

      // Current alpha
      TScalar phi_alpha_fval;
      TScalar phi_alpha_grad;

      // Find a stepsize interval satisfying the strong Wolfe conditions for a
      // given iterate x and gradient gx and descent direction d:
      //   1. Armijo: f(x + alpha d) <= f(x) + c1 alpha gx' d,
      //   2. Curvature: |nabla_alpha f(x + alpha d)| <= c2 |gx' d|.
      unsigned int max_test = 200;
      for (unsigned int n = 0; true; ++n) 
      {
	 if (alpha <= alpha_min) 
            break;
	 else if (alpha >= alpha_max || n >= max_test) 
         {
	    alpha = std::numeric_limits<TScalar>::signaling_NaN();
	    break;
	 }
	 Evaluate(alpha, phi_alpha_fval, phi_alpha_grad);

	 // If Armijo condition is violated: zoom, as a point satisfying the
	 // Wolfe condition must exist in [alpha_{i-1}, alpha_i].
	 if (phi_alpha_fval > (x0_fval + 1.0e-4 * alpha * x0_phi_grad)) 
         {
	    alpha *= fa_shrink;
	    continue;
	 }

	 // The Armijo condition is satisfied.  If in addition the curvature
	 // condition ("function must be flat around alpha") is satisfied, then
	 // we found a point satisfying the Wolfe conditions.
	 if (phi_alpha_grad < 0.9 * x0_phi_grad) 
         {
	    alpha *= fa_enlarge;
	    continue;
	 }
	 break;
      }
      return (evaluation_count);
   }

   unsigned int ComputeStepLengthUpdate(TVector& x_out, TVector& grad_out, 
      TScalar& fval_out, TScalar& alpha)
   {
      unsigned int ecount = ComputeStepLength(alpha);

      if (alpha != alpha)	// FIXME: VC++ does not support C99
         return (ecount);

      // A valid step size has been computed
      if (xalpha_val != alpha) 
      {
	 // Is not up-to-date, update
	 TScalar d1, d2;	// dummy
	 Evaluate(alpha, d1, d2);
      }
      assert(xalpha_val == alpha);
      x_out = xalpha;
      grad_out = xalphagrad;
      fval_out = xalphaobj;

      return (ecount);
   }

private:
   Func& prob;
   const TVector x0;
   const TVector x0_grad;
   const TVector H_grad;
   TScalar x0_fval;
   TScalar x0_phi_grad;

   unsigned int evaluation_count;

   // All information from the function evaluation if xalpha_val != nan
   std::vector<TScalar> xalpha;
   std::vector<TScalar> xalphagrad;
   TScalar xalphaobj;
   TScalar xalpha_val;

   void Evaluate(TScalar alpha, TScalar& phi_fval, TScalar& phi_grad)
   {
      // x(alpha) = x - alpha*p
      xalpha = x0 - alpha * H_grad;
      std::fill(xalphagrad.begin(), xalphagrad.end(), TScalar(0));

      // Evaluate phi(alpha) and derivative phi'(alpha)
      // phi(alpha) = f(x_k - alpha H_grad)
      // phi'(alpha) = -H_grad' \nabla_x f(x_k - alpha H_grad)
      phi_fval = prob(xalpha.begin(), xalphagrad.begin());
      xalphaobj = phi_fval;	// save phi(alpha)
      xalpha_val = alpha;	// save alpha

      // Univariate derivative is projection onto ascent direction
      phi_grad = -dot(xalphagrad, H_grad);
      evaluation_count++;
   }
};

template <typename TScalar>
inline static void show(unsigned int iter, TScalar obj, TScalar grad_norm, size_t lbfgs_mem_size, unsigned int ls_evals)
{
   if (iter % 20 == 0) {
	 std::cout << std::endl;
	 std::cout << "  iter      objective      |grad|   "
		  << "mem     ls#" << std::endl;
   }
   std::ios_base::fmtflags original_format = std::cout.flags();
   std::streamsize original_prec = std::cout.precision();

   // Iteration
   std::cout << std::setiosflags(std::ios::left)
         << std::setiosflags(std::ios::adjustfield)
         << std::setw(6) << iter << "  ";
   std::cout << std::resetiosflags(std::ios::fixed);

   // Objective function
   std::cout << std::setiosflags(std::ios::scientific)
         << std::setprecision(5)
         << std::setiosflags(std::ios::left)
         << std::setiosflags(std::ios::showpos)
         << std::setw(7) << obj << "   ";
   // Gradient norm
   std::cout << std::setiosflags(std::ios::scientific)
         << std::setprecision(2)
         << std::resetiosflags(std::ios::showpos)
         << std::setiosflags(std::ios::left) << grad_norm;
   // LBFGS memory size
   std::cout << std::setiosflags(std::ios::left)
         << std::setiosflags(std::ios::adjustfield)
         << std::setw(6) << lbfgs_mem_size << "  ";
   std::cout << std::setiosflags(std::ios::left)
         << std::setiosflags(std::ios::adjustfield)
         << std::setw(6) << ls_evals << "  ";
   std::cout << std::endl;

   std::cout.precision(original_prec);
   std::cout.flags(original_format);
}

template <typename TScalar, typename TVector>
class hessian
{
public:
   hessian(size_t m) : m(m) {}

   size_t size() const { return H_k.size(); }

   // grad := -(H_k * grad)
   void update_gradient(TVector& grad) const
   {
      size_t k = H_k.size();
      std::vector<TScalar> alphas(H_k.size());
      {
         auto alpha_i = alphas.begin();
         for (auto i = H_k.begin(); i != H_k.end(); ++i, ++alpha_i)
         {
            *alpha_i = dot(i->s, grad) / dot(i->y, i->s);
            grad -= *alpha_i * i->y;
         }
      }
      TScalar gamma_k = 1;
      if (!H_k.empty())
      {
         const auto& front = H_k.front();
         gamma_k = dot(front.s, front.y) / dot(front.y, front.y);
      }
      grad *= gamma_k;
      {
         auto alpha_i = alphas.rbegin();
         for (auto i = H_k.rbegin(); i != H_k.rend(); ++i, ++alpha_i)
            grad += i->s * (*alpha_i - i->rho * dot(i->y, grad));
      }
   }

   void update_hessian(const TVector& s_k, const TVector& y_k)
   {
      // Heuristically ensure stability by ignoring unstable updates.
      // As to what entails 'unstable' there exist different opinions.
      TScalar ys_p = dot(s_k, y_k);
      if (ys_p >= 1.0e-12 * dot(y_k, y_k))
      {
         // Remove old element from the lbfgs memory, if necessary
         if (H_k.size() >= m)
            H_k.pop_back();
         element el = {s_k, y_k, 1 / ys_p};
         H_k.push_front(el);
      } 
   }

protected:
   size_t m;
   struct element
   {
      TVector s;
      TVector y;
      TScalar rho;
   };
   std::deque<element> H_k;
};

template <typename TScalar, typename TVector>
bool IsDegenerateGradient(const TVector& gradprev, const TVector& grad)
{
   // Check cosine angle between gradient and transformed gradient
   TScalar x0_phi_grad = dot(grad, gradprev);
   TScalar cos_a = x0_phi_grad / (length(gradprev) * length(grad));
   if (cos_a <= -1.0e-8) 
      throw std::exception("LBFGS approximation lost psd!");
   else if (cos_a <= 1.0e-7) 
   {
      // Numerical issues, degenerate true Hessian or converged.
      return true;
   }

   // if (std::isnan(x0_phi_grad)) // FIXME: VC++ does not support C99
   if (x0_phi_grad != x0_phi_grad) 
   {	
      // WARNING: gradient or perturbed gradient NaN
      assert(false);
      return true;
   }
   // Check it is a descent direction
   if (x0_phi_grad <= -1.0e-8) 
      throw std::exception("LBFGS approximation lost psd!");
   else if (x0_phi_grad <= 1.0e-10) 
   {
      // Numerical issues, degenerate true Hessian or converged.
      return true;
   }
   return false;
}
}

// L-BFGS
template <typename TScalar, typename Func>
std::vector<TScalar> LBFGS(Func& prob,
	unsigned int dim, std::vector<TScalar>& x, TScalar conv_tol, unsigned int max_iter,
	unsigned int lbfgs_m, bool verbose = false) 
{
   if (x.size() != dim)
      throw std::exception("LBFGS: x size doesn't match dim.");

   // Gradient, last gradient and alpha value required for BB iteration
   typedef std::vector<TScalar> TVector;
   TVector grad(dim, 0.0);
   TVector grad_last(dim, 0.0);
   TVector xprev(dim);
   TVector gradprev(dim);

   // List of previous s,y,rho_i
   _LBFGS::hessian<TScalar, TVector> H_k(lbfgs_m);

   std::vector<TScalar> objectives;
   objectives.push_back(prob(x.begin(), grad.begin()));
   TScalar grad_norm = ts::length(grad);
   unsigned int ls_evals = 0;
   if (verbose)
      _LBFGS::show<TScalar>(0, objectives.back(), grad_norm, H_k.size(), ls_evals);
   TScalar alpha = 1;
   const TScalar alpha_tol = static_cast<TScalar>(1e-12);
   for (unsigned int iter = 0; iter < max_iter && grad_norm > conv_tol && alpha > alpha_tol; ++iter) 
   {
      // Insert differential information into Hessian approximation
      if (iter > 0) 
         H_k.update_hessian(x - xprev, grad - gradprev);

      // Save current iterate for next update
      xprev = x;      
      gradprev = grad;

      // Adjust gradient direction
      H_k.update_gradient(grad);

      // Check for numerical stability conditions
      if (_LBFGS::IsDegenerateGradient<TScalar>(gradprev, grad))
         break;

      // Perform linesearch in descent direction
      TScalar alpha = 1;
      TScalar obj = objectives.back();
      _LBFGS::SimpleLineSearch<TScalar, TVector, Func> linesearch(prob, x, gradprev, grad, obj);
      // Be very careful on the first step
      if (iter == 0) 
      {
         TScalar len = length(grad);
         if (len > 1)
            alpha = 1 / len;
         alpha = std::max(static_cast<TScalar>(1.0e-12), alpha);
      }
      ls_evals = linesearch.ComputeStepLengthUpdate(x, grad, obj, alpha);

      if (alpha != alpha)       // if ((boost::math::isnan)(alpha))
      {
         // Line search failed - revert
         x = xprev;
         grad = gradprev;
         break;
      }
      // Successful line search with up-to-date step
      objectives.push_back(obj);
      grad_norm = length(grad);
      if (verbose)
         _LBFGS::show<TScalar>(iter + 1, objectives.back(), grad_norm, H_k.size(), ls_evals);
   }
   return objectives;
}

}
}