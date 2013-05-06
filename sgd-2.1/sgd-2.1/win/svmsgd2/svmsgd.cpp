// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2007- Leon Bottou

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include "assert.h"
#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include "loss.h"
#include "data.h"

using namespace std;

// hello


// ---- Loss function

// Compile with -DLOSS=xxxx to define the loss function.
// Loss functions are defined in file loss.h)
#ifndef LOSS
# define LOSS LogLoss
#endif

// ---- Bias term

// Compile with -DBIAS=[1/0] to enable/disable the bias term.
// Compile with -DREGULARIZED_BIAS=1 to enable regularization on the bias term

#ifndef BIAS
# define BIAS 1
#endif
#ifndef REGULARIZED_BIAS
# define REGULARIZED_BIAS 0
#endif

ofstream out;

// ---- Plain stochastic gradient descent

class SvmSgd
{
public:
  SvmSgd(int dim, double lambda, double eta0  = 0);
  void renorm();
  double wnorm();
  double testOne(const SVector &x, double y, double *ploss, double *pnerr);
  void trainOne(const SVector &x, double y, double eta);
public:
  void train(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
public:
  double evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
  void determineEta0(int imin, int imax, const xvec_t &x, const yvec_t &y);
private:
  double  lambda;
  double  eta0;
  FVector w;
  double  wDivisor;
  double  wBias;
  double  t;
};

/// Constructor
SvmSgd::SvmSgd(int dim, double lambda, double eta0)
  : lambda(lambda), eta0(eta0), 
    w(dim), wDivisor(1), wBias(0),
    t(0)
{
}

/// Renormalize the weights
void
SvmSgd::renorm()
{
  if (wDivisor != 1.0)
    {
      w.scale(1.0 / wDivisor);
      wDivisor = 1.0;
    }
}

/// Compute the norm of the weights
double
SvmSgd::wnorm()
{
  double norm = dot(w,w) / wDivisor / wDivisor;
#if REGULARIZED_BIAS
  norm += wBias * wBias
#endif
  return norm;
}

/// Compute the output for one example.
double
SvmSgd::testOne(const SVector &x, double y, double *ploss, double *pnerr)
{
  double s = dot(w,x) / wDivisor + wBias;
  if (ploss)
    *ploss += LOSS::loss(s, y);
  if (pnerr)
    *pnerr += (s * y <= 0) ? 1 : 0;
  return s;
}

/// Perform one iteration of the SGD algorithm with specified gains
void
SvmSgd::trainOne(const SVector &x, double y, double eta)
{
  double s = dot(w,x) / wDivisor + wBias;
  // update for regularization term
  wDivisor = wDivisor / (1 - eta * lambda);
  if (wDivisor > 1e5) renorm();
  // update for loss term
  double d = LOSS::dloss(s, y);
  if (d != 0)
    w.add(x, eta * d * wDivisor);
  // same for the bias
#if BIAS
  double etab = eta * 0.01;
#if REGULARIZED_BIAS
  wBias *= (1 - etab * lambda);
#endif
  wBias += etab * d;
#endif
}


/// Perform a training epoch
void 
SvmSgd::train(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  assert(eta0 > 0);
  for (int i=imin; i<=imax; i++)
    {
      double eta = eta0 / (1 + lambda * eta0 * t);
      trainOne(xp.at(i), yp.at(i), eta);
      t += 1;
    }
  cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
  cout << " wBias=" << wBias;
#endif
  cout << endl;
}

/// Perform a test pass
void 
SvmSgd::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  double nerr = 0;
  double loss = 0;
  for (int i=imin; i<=imax; i++)
    testOne(xp.at(i), yp.at(i), &loss, &nerr);
  nerr = nerr / (imax - imin + 1);
  loss = loss / (imax - imin + 1);
  double cost = loss + 0.5 * lambda * wnorm();
  cout << prefix 
       << "Loss=" << setprecision(12) << loss
       << " Cost=" << setprecision(12) << cost 
       << " Misclassification=" << setprecision(4) << 100 * nerr << "%." 
       << endl;
}

/// Perform one epoch with fixed eta and return cost

double 
SvmSgd::evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta)
{
  SvmSgd clone(*this); // take a copy of the current state
  assert(imin <= imax);
  for (int i=imin; i<=imax; i++)
    clone.trainOne(xp.at(i), yp.at(i), eta);
  double loss = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    clone.testOne(xp.at(i), yp.at(i), &loss, 0);
  loss = loss / (imax - imin + 1);
  cost = loss + 0.5 * lambda * clone.wnorm();
  // cout << "Trying eta=" << eta << " yields cost " << cost << endl;
  return cost;
}

void 
SvmSgd::determineEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp)
{
  const double factor = 2.0;
  double loEta = 1;
  double loCost = evaluateEta(imin, imax, xp, yp, loEta);
  double hiEta = loEta * factor;
  double hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
  if (loCost < hiCost)
    while (loCost < hiCost)
      {
        hiEta = loEta;
        hiCost = loCost;
        loEta = hiEta / factor;
        loCost = evaluateEta(imin, imax, xp, yp, loEta);
      }
  else if (hiCost < loCost)
    while (hiCost < loCost)
      {
        loEta = hiEta;
        loCost = hiCost;
        hiEta = loEta * factor;
        hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
      }
  eta0 = loEta;
  cout << "# Using eta0=" << eta0 << endl;
}


// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 5;
int maxtrain = -1;


void
usage(const char *progname)
{
  const char *s = ::strchr(progname,'/');
  progname = (s) ? s + 1 : progname;
  cerr << "Usage: " << progname << " [options] trainfile [testfile]" << endl
       << "Options:" << endl;
#define NAM(n) "    " << setw(16) << left << n << setw(0) << ": "
#define DEF(v) " (default: " << v << ".)"
  cerr << NAM("-lambda x")
       << "Regularization parameter" << DEF(lambda) << endl
       << NAM("-epochs n")
       << "Number of training epochs" << DEF(epochs) << endl
       << NAM("-dontnormalize")
       << "Do not normalize the L2 norm of patterns." << endl
       << NAM("-maxtrain n")
       << "Restrict training set to n examples." << endl;
#undef NAM
#undef DEF
  ::exit(10);
}

void
parse(int argc, const char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
        {
          if (trainfile == 0)
            trainfile = arg;
          else if (testfile == 0)
            testfile = arg;
          else
            usage(argv[0]);
        }
      else
        {
          while (arg[0] == '-') 
            arg += 1;
          string opt = arg;
          if (opt == "lambda" && i+1<argc)
            {
              lambda = atof(argv[++i]);
              assert(lambda>0 && lambda<1e4);
            }
          else if (opt == "epochs" && i+1<argc)
            {
              epochs = atoi(argv[++i]);
              assert(epochs>0 && epochs<1e6);
            }
          else if (opt == "dontnormalize")
            {
              normalize = false;
            }
          else if (opt == "maxtrain" && i+1 < argc)
            {
              maxtrain = atoi(argv[++i]);
              assert(maxtrain > 0);
            }
          else
            {
              cerr << "Option " << argv[i] << " not recognized." << endl;
              usage(argv[0]);
            }

        }
    }
  if (! trainfile)
    usage(argv[0]);
}

void 
config(const char *progname)
{
  cout << "# Running: " << progname;
  cout << " -lambda " << lambda;
  cout << " -epochs " << epochs;
  if (! normalize) cout << " -dontnormalize";
  if (maxtrain > 0) cout << " -maxtrain " << maxtrain;
  cout << endl;
#define NAME(x) #x
#define NAME2(x) NAME(x)
  cout << "# Compiled with: "
       << " -DLOSS=" << NAME2(LOSS)
       << " -DBIAS=" << BIAS
       << " -DREGULARIZED_BIAS=" << REGULARIZED_BIAS
       << endl;
}

//////////////////////////////////////////////////////////////////////////
// wrapper for multi-class svm
//////////////////////////////////////////////////////////////////////////
int GetMaxLabel(const yvec_t& ytrain)
{
	// labels must be from 0
	double max_label = 0;
	for (size_t i=0; i<ytrain.size(); i++)
	{
		if(ytrain[i] > max_label)
			max_label = ytrain[i];
	}

	return (int)max_label;
}

bool SetupLabelsForClass(const yvec_t& ytrain_in, yvec_t& ytrain_out, int pos_label)
{
	ytrain_out = ytrain_in;
	for(size_t i=0; i<ytrain_out.size(); i++)
	{
		if(ytrain_out[i] == pos_label)
			ytrain_out[i] = 1;
		else
			ytrain_out[i] = -1;
	}
	
	return true;
}

bool train_Multiclass(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix, 
					  double label_num, vector<SvmSgd>& svms)
{
	if(svms.size() != label_num)
		return false;

	// train svm for each class
	for(int i=0; i<label_num; i++)
	{
		yvec_t new_labels;
		SetupLabelsForClass(yp, new_labels, i);

		// train
		svms[i].train(imin, imax, xp, new_labels, prefix);
	}

	return true;
}

double testOne_Multiclass(vector<SvmSgd>& svms, const SVector &x, double y, double *pnerr)
{
	double max_s = 0;
	double max_cls_id = -1;
	
	for(size_t i=0; i<svms.size(); i++)
	{
		double ploss=0, nerr=0;
		double s = svms[i].testOne(x, y, &ploss, &nerr);
		if(max_cls_id == -1)
		{
			max_s = s;
			max_cls_id = i;
		}
		else
		{
			if(s > max_s)
			{
				max_s = s;
				max_cls_id = i;
			}
		}
	}

	if( y != max_cls_id )
		*pnerr += 1;

	return max_cls_id;
}


bool test_Multiclass(vector<SvmSgd>& svms, int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char* prefix)
{
	cout << prefix << "Testing multiclass on [" << imin << ", " << imax << "]." << endl;
	out << prefix << "Testing multiclass on [" << imin << ", " << imax << "]." << endl;
	if(imin > imax)
		return false;

	double nerr = 0;
	double loss = 0;
	for (int i=imin; i<=imax; i++)
		testOne_Multiclass(svms, xp.at(i), yp.at(i), &nerr);
	nerr = nerr / (imax - imin + 1);

	cout << " Misclassification=" << setprecision(4) << 100 * nerr << "%." 
		<< endl;
	out << " Misclassification=" << setprecision(4) << 100 * nerr << "%." 
		<< endl;

	return true;
}


// generate 0:n-1
vector<int> generateArray(int n)
{
	vector<int> nums(n);
	for(int i=0; i<n; i++)
		nums[i] = i;

	return nums;
}

//////////////////////////////////////////////////////////////////////////


// --- main function

int dims;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;




int main(int argc, const char **argv)
{
  parse(argc, argv);
  config(argv[0]);
  if (trainfile)
    load_datafile(trainfile, xtrain, ytrain, dims, normalize, maxtrain);
  if (testfile)
    load_datafile(testfile, xtest, ytest, dims, normalize);
  cout << "# Number of features " << dims << "." << endl;

  // prepare svm
  int imin = 0;
  int imax = xtrain.size() - 1;
  int tmin = 0;
  int tmax = xtest.size() - 1;

  string outfile = "lbp_task10_super.txt";
  out.open(outfile.c_str());
  if(!out.is_open())
	  return -1;

  int max_train_label;
  max_train_label = GetMaxLabel(ytrain);

  // organize all data into class-wise subgroups
  map<int, xvec_t> class_samples;
	// training
	for(size_t j=0; j<ytrain.size(); j++)
	{
		int cur_label = (int)ytrain[j];
		class_samples[cur_label].push_back(xtrain[j]);
	}
	// testing
	for(size_t j=0; j<ytest.size(); j++)
	{
		int cur_label = (int)ytest[j];
		class_samples[cur_label].push_back(xtest[j]);
	}



  // shuffle dataset to generate multiple random training set and testing set
  int trial_num = 20;
  std::srand ( unsigned ( std::time(0) ) );
  for(int i=0; i<trial_num; i++)
  {
	  cout<<endl<<"Trial: "<<i<<endl<<endl;
	  out<<endl<<"Trial: "<<i<<endl<<endl;

	  // construct new training and testing set
	  xtrain.clear();
	  ytrain.clear();
	  xtest.clear();
	  ytest.clear();

	  for(map<int, xvec_t>::iterator pi=class_samples.begin(); pi!=class_samples.end(); pi++)
	  {
		  int cur_label = pi->first;
		  const xvec_t& cur_samps = pi->second;
		  vector<int> ids = generateArray(cur_samps.size());
		  // random shuffle
		  std::random_shuffle(ids.begin(), ids.end());

		  // 80% as training samples; 20% as testing samples
		  for(int k=0; k<ids.size(); k++)
		  {
			  if(k<ids.size()*0.8)
			  {
				  xtrain.push_back(cur_samps[ids[k]]);
				  ytrain.push_back(cur_label);
			  }
			  else
			  {
				  xtest.push_back(cur_samps[ids[k]]);
				  ytest.push_back(cur_label);
			  }
		  }
	  }

	#ifndef MULTICLASS

	  SvmSgd svm(dims, lambda);
	  Timer timer;
	  // determine eta0 using sample
	  int smin = 0;
	  int smax = imin + min(1000, imax);
	  timer.start();
	  svm.determineEta0(smin, smax, xtrain, ytrain);
	  timer.stop();
	  // train
	  for(int i=0; i<epochs; i++)
		{
		  cout << "--------- Epoch " << i+1 << "." << endl;
		  timer.start();
		  svm.train(imin, imax, xtrain, ytrain);
		  timer.stop();
		  cout << "Total training time " << setprecision(6) 
			   << timer.elapsed() << " secs." << endl;
		  svm.test(imin, imax, xtrain, ytrain, "train: ");
		  if (tmax >= tmin)
			svm.test(tmin, tmax, xtest, ytest, "test:  ");
		}

	#else

	  // determine eta0 using sample
	  int smin = 0;
	  int smax = imin + min(1000, imax);
  
	  vector<SvmSgd> svms;
	  for(size_t i=0; i<max_train_label+1; i++)
	  {
		  SvmSgd svm(dims, lambda);
		  svm.determineEta0(smin, smax, xtrain, ytrain);
		  svms.push_back(svm);
	  }

	  // train
	  Timer timer;
	  for(int i=0; i<epochs; i++)
	  {
		  cout << "--------- Epoch " << i+1 << "." << endl;
		  out << "--------- Epoch " << i+1 << "." << endl;
		  timer.start();
		  train_Multiclass(imin, imax, xtrain, ytrain, "Multiclass train: ", max_train_label+1, svms);
		  timer.stop();
		  cout << "Total training time " << setprecision(6) 
			  << timer.elapsed() << " secs." << endl;
		  out << "Total training time " << setprecision(6) 
			  << timer.elapsed() << " secs." << endl;
		  test_Multiclass(svms, imin, imax, xtrain, ytrain, "Train: ");
		  if (tmax >= tmin)
			  test_Multiclass(svms, tmin, tmax, xtest, ytest, "Test: ");
	  }
	#endif

  }


  getchar();
  return 0;
}


