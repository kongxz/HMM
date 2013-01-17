//
//  BaseHMM.cpp
//
//  Created by Xiangzhen Kong on 12-6-27.
//  Copyright (c) 2012Äê __MyCompanyName__. All rights reserved.
//

#include "BaseHMM.h"


BaseHMM::BaseHMM(int stateNum, ObservationPDInterface *b)
	: m_N(stateNum)
	,m_B(b)
{
	m_A.resize(m_N, m_N);
	m_pi.resize(m_N);
	init();
}


BaseHMM::~BaseHMM(void)
{
}

void BaseHMM::init()
{   
    for(int i=0; i<m_N; i++)
    {
        m_pi[i] = 1.0/m_N;
        for(int j=0; j<m_N; j++)
        {
			m_A.set(i, j, 1.0/m_N);
        }
    }
    
}

float BaseHMM::evaluate(std::vector<SymbolInterface*> observation)
{
 //   float *alpha = new float[T * m_N];
 //   computeAlpha(alpha, observableStates, T);
 //   
 //   float prob = 0.0;
	//float *alphaT = alpha + (T-1)*m_N;
 //   for(int j=0; j<m_N; j++)
 //   {
 //       prob += alphaT[j];
 //   }
 //   
	//delete [] alpha;

 //   return prob;
	int T = observation.size();

	Matrix<float> alpha = computeAlpha(observation);
	float prob = 0.0;
	
	for(int j=0; j<m_N; j++)
	{
		prob += alpha.get(T-1, j);
	}

	return prob;
}

//void HMM::decode(int *observableStates, int number, int **hiddenStates)
std::vector<int> BaseHMM::decode(std::vector<SymbolInterface*> observation)
{
	std::vector<int> hiddenStates;

	int T = observation.size();
	Matrix<float> delta(T, m_N);
	Matrix<int> phi(T, m_N);
	for(int i=0; i<m_N; i++)
	{
		float v = m_pi[i]*m_B->observationProbability(i, observation[0]);
		delta.set(0, i, v);
	}

	for(int t=1; t<T; t++)
	{
		for(int j=0; j<m_N; j++)
		{
			float maxP = 0.0;
			int pointer = 0;
			for(int i=0; i<m_N; i++)
			{
				float p = delta.get(t-1, i)*m_A.get(i,j)*m_B->observationProbability(j,observation[t]);
				if(maxP < p)
				{
					maxP = p;
					pointer = i;
				}
			}
			delta.set(t, j, maxP);
			phi.set(t, j, pointer);
		}
	}
	
	float prob = 0.0;
	int pointer = 0;
	for(int j=0; j<m_N; j++)
	{
		if(prob < delta.get(T-1, j))
		{
			prob = delta.get(T-1, j);
			pointer = j;
		}
	}
    
	for(int t= T-1; t>=0; t--)
	{
		hiddenStates[t] = pointer;
		pointer = phi.get(t, pointer);
	}

	return hiddenStates;

    //float *delta = new float[number * m_N];
    //int *phi = new int[number*m_N];
    //for (int i=0; i<m_N; i++) 
    //{
    //    delta[i]= m_pi[i]*m_B[m_M*i+observableStates[0]];
    //}
    //
    //for (int t=1; t<number; t++) 
    //{
    //    for (int j=0; j<m_N; j++)
    //    {
    //        float maxP = 0;
    //        int  pointer = 0;
    //        for (int i=0; i<m_N; i++) 
    //        {
    //            float p = delta[(t-1)*m_N+i]*m_A[i*m_N+j]*m_B[j*m_M+observableStates[t]];
    //            if(maxP < p)
    //            {
    //                maxP = p;
    //                pointer = i;
    //            }
    //        }
    //        delta[t*m_N+j] = maxP;
    //        phi[t*m_N+j] = pointer;
    //    }
    //}
    //
    //float prob = 0;
    //int pointer = 0;
    //for(int j=0; j<m_N; j++)
    //{
    //    if (prob < delta[(number-1)*m_N+j])
    //    {
    //        prob = delta[(number-1)*m_N+j];
    //        pointer = j;
    //    }
    //}
    //
    //
    //for(int t = number-1; t>=0; t--)
    //{
    //    (*hiddenStates)[t] = pointer;
    //    pointer = phi[t*m_N+pointer];
    //}

}

//void HMM::train(int *observableStates, int T)
void BaseHMM::train(std::vector<SymbolInterface*> observation)
{
	int T = observation.size();

	// compute alpha
	Matrix<float> alpha = computeAlpha(observation);

	// compute beta
	Matrix<float> beta = computeBeta(observation);

	Matrix<float> gamma(T, m_N);
	std::vector<Matrix<float>> xi = computeXi(observation, gamma, alpha, beta);

	// update pi
	for(int i=0; i<m_N; i++)
	{
		m_pi[i] = gamma.get(0, i);
	}

	// update A
	for(int i=0; i<m_N; i++)
	{
		float sumGama = 0.0;
        for(int t=0; t<T-1; t++)
        {
			sumGama += gamma.get(t, i);
        }
        
        for(int j=0; j<m_N; j++)
        {
            float sumXi = 0.0;
            for(int t=0; t<T-1; t++)
            {
				sumXi += xi[t].get(i, j);
            }
			m_A.set(i, j, sumXi/sumGama);
        }
	}

	// update B
  //  for(int i=0; i<m_N; i++)
  //  {
  //      float sumGama = 0.0;
  //      for(int t=0; t<T; t++)
  //      {
		//	sumGama += gamma.get(t, i);
  //      }
  //      
		//for(int k=0; k<m_M; k++)
		//{
		//	float sumGamaK = 0.0;
		//	for(int t=0; t<T; t++)
		//	{
		//		if(observableStates[t]==k)
		//		{
		//			sumGamaK += gamma[t*m_N+i];
		//		}
		//	}
		//	m_B[i*m_N+k] = sumGamaK/sumGama;
		//}
  //  }

 //   // compute alpha
 //   float *alpha = new float[T * m_N];
 //   computeAlpha(alpha, observableStates, T);
 //   
 //   // compute beta
 //   float *beta = new float[T * m_N];
 //   computeBeta(beta, observableStates, T);    

 //   float *xi = new float[T * m_N * m_N];
 //   float *gamma = new float[T * m_N];
 //   computeXi(xi, gamma, observableStates, T, alpha, beta);
 //   
 //   //float *gamma = new float[T * m_N];
 //   //computeGamma(gamma, observableStates, T, alpha, beta);
 //   
 //   // update pi
 //   for (int i=0; i<m_N; i++)
 //   {
 //       m_pi[i] = gamma[i];
 //   }
 //   
 //   // update A
 //   for(int i=0; i<m_N; i++)
 //   {
 //       float sumGama = 0.0;
 //       for(int t=0; t<T-1; t++)
 //       {
 //           sumGama += gamma[t*m_N+i];
 //       }
 //       
 //       for(int j=0; j<m_N; j++)
 //       {
 //           float sumXi = 0.0;
 //           for(int t=0; t<T-1; t++)
 //           {
 //               sumXi += xi[t*m_N*m_N+i*m_N+j];
 //           }
 //           m_A[i*m_N+j] = sumXi/sumGama;
 //       }
 //   }
 //   
 //   // update B
 //   for(int i=0; i<m_N; i++)
 //   {
 //       float sumGama = 0.0;
 //       for(int t=0; t<T; t++)
 //       {
 //           sumGama += gamma[t*m_N+i];
 //       }
 //       
	//	for(int k=0; k<m_M; k++)
	//	{
	//		float sumGamaK = 0.0;
	//		for(int t=0; t<T; t++)
	//		{
	//			if(observableStates[t]==k)
	//			{
	//				sumGamaK += gamma[t*m_N+i];
	//			}
	//		}
	//		m_B[i*m_N+k] = sumGamaK/sumGama;
	//	}
 //   }

	//delete [] alpha;
	//delete [] beta;
	//delete [] xi;
}

Matrix<float> BaseHMM::computeAlpha(std::vector<SymbolInterface*> &observation)
{
	int T = observation.size();
	Matrix<float> alpha(T, m_N);
    for (int i=0; i<m_N; i++) 
    {
        //alpha[i]= m_pi[i]*m_B[m_M*i+observableStates[0]];
		alpha.set(0, i, m_B->observationProbability(i, observation[0]));
    }
    
    for (int t=1; t<T; t++) 
    {
        for (int j=0; j<m_N; j++)
        {
            float sum = 0;
            for (int i=0; i<m_N; i++) 
            {
                //sum += alpha[(t-1)*m_N+i]*m_A[i*m_N+j]*m_B[j*m_M+observableStates[t]];
				sum += alpha.get(t-1, i)*m_A.get(i, j)*m_B->observationProbability(j, observation[t]);
            }
            //alpha[t*m_N+j] = sum;
			alpha.set(t, j, sum);
        }
    }

	return alpha;
}

Matrix<float> BaseHMM::computeBeta(std::vector<SymbolInterface*> &observation)
{
	int T = observation.size();
	Matrix<float> beta(T, m_N);

    for (int i=0; i<m_N; i++) 
    {
        //beta[(T-1)*m_N+i]= 1;
		beta.set(T-1, i, 1.0);
    }
    
    for (int t=T-2; t>=0; t--) 
    {
        for (int i=0; i<m_N; i++)
        {
            float sum = 0;
            for (int j=0; j<m_N; j++) 
            {
                //sum += beta[(t+1)*m_N+j]*m_A[i*m_N+j]*m_B[j*m_M+observableStates[t+1]];
				sum += beta.get(t+1, j)*m_A.get(i, j)*m_B->observationProbability(j, observation[t+1]);
            }
            //beta[t*m_N+i] = sum;
			beta.set(t, i, sum);
        }
    }

	return beta;
}

//void HMM::computeGamma(float *&gamma, int* observableStates, int T, float *alpha, float *beta)
Matrix<float> BaseHMM::computeGamma(std::vector<SymbolInterface*> &observation, Matrix<float> &alpha, Matrix<float> &beta)
{
	int T = observation.size();
	Matrix<float> gamma(T, m_N);

	for(int t=0; t<T; t++)
    {
        //float *gammaT = gamma+t*m_N;
        float sum = 0.0;
        for (int i=0; i<m_N; i++)
        {
            /*gammaT[i] = alpha[t*m_N+i]*beta[t*m_N+i];
            sum += gammaT[i];*/
			float v = alpha.get(t, i)*beta.get(t, i);
			gamma.set(t, i, v);
			sum += v;
        }
        
        for(int i=0; i<m_N; i++)
        {
            //gammaT[i] /= sum;
			gamma.set(t, i, gamma.get(t, i)/sum);
        }
    }

	return gamma;
}

//void HMM::computeXi(float *&Xi, float *&gamma, int* observableStates, int T, float *alpha, float *beta)
std::vector<Matrix<float>> BaseHMM::computeXi(std::vector<SymbolInterface*> &observation, Matrix<float> &gamma, Matrix<float> &alpha, Matrix<float> &beta)
{
	int T = observation.size();
	//Matrix<float> beta(T, m_N);
	std::vector<Matrix<float>> Xi;

	for (int t=0; t<T-1; t++)
    {
     /*   float *xiT = Xi+(t*m_N*m_N);
		float *gammaT = gamma + t*m_N;*/
		
		Matrix<float> xiT(m_N, m_N);

        float sum = 0.0;
        for(int i=0; i<m_N; i++)
        {
		    for(int j=0; j<m_N; j++)
            {
                /*xiT[i*m_N+j] = alpha[t*m_N+i]*m_A[i*m_N+j]*beta[(t+1)*m_N+j]*m_B[j*m_N+observableStates[t+1]];
                sum += xiT[i*m_N+j];*/
				float v = alpha.get(t, i)*m_A.get(i, j)*beta.get(t+1, j)*m_B->observationProbability(j, observation[t+1]);
				xiT.set(i, j, v);
				sum += v;
            }
        }
        
        for(int i=0; i<m_N; i++)
        {
			//gammaT[i] = 0.0;
			gamma.set(t, i, 0.0);
            for(int j=0; j<m_N; j++)
            {
                /*xiT[i*m_N+j] /= sum;
				gammaT[i] += xiT[i*m_N+j];*/
				xiT.set(i,j, xiT.get(i,j)/sum);
				gamma.set(t, i, xiT.get(i, j));
            }
        }

		Xi.push_back(xiT);
    }
	return Xi;
}








