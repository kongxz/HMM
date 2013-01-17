//
//  HMM.cpp
//  HMM
//
//  Created by Xiangzhen Kong on 12-6-27.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//

#include "HMM.h"

#include <iostream>

HMM::HMM(int hidden, int observable)
:m_N(hidden)
,m_M(observable)
{
    init();
}

HMM::~HMM()
{
    if (m_pi != NULL) 
    {
        delete [] m_pi;
        m_pi = NULL;
    }
    
    if (m_A != NULL) 
    {
        delete [] m_A;
        m_A = NULL;
    }
    
    if (m_B != NULL) 
    {
        delete [] m_B;
        m_B = NULL;
    }
}

void HMM::init()
{
    m_pi = new float[m_N];
    m_A = new float[m_N*m_N];
    m_B = new float[m_N*m_M];
    
    for(int i=0; i<m_N; i++)
    {
        m_pi[i] = 1.0/m_N;
        for(int j=0; j<m_N; j++)
        {
            m_A[i*m_N + j] = 1.0/m_N;
        }
        
        for(int j=0; j<m_M; j++)
        {
            m_B[i*m_M + j] = 1.0/m_M;
        }
    }
    
}

float HMM::evaluate(int *observableStates, int T)
{
    float *alpha = new float[T * m_N];
    computeAlpha(alpha, observableStates, T);
    
    float prob = 0.0;
	float *alphaT = alpha + (T-1)*m_N;
    for(int j=0; j<m_N; j++)
    {
        prob += alphaT[j];
    }
    
	delete [] alpha;

    return prob;
}

void HMM::decode(int *observableStates, int number, int **hiddenStates)
{
    float *delta = new float[number * m_N];
    int *phi = new int[number*m_N];
    for (int i=0; i<m_N; i++) 
    {
        delta[i]= m_pi[i]*m_B[m_M*i+observableStates[0]];
    }
    
    for (int t=1; t<number; t++) 
    {
        for (int j=0; j<m_N; j++)
        {
            float maxP = 0;
            int  pointer = 0;
            for (int i=0; i<m_N; i++) 
            {
                float p = delta[(t-1)*m_N+i]*m_A[i*m_N+j]*m_B[j*m_M+observableStates[t]];
                if(maxP < p)
                {
                    maxP = p;
                    pointer = i;
                }
            }
            delta[t*m_N+j] = maxP;
            phi[t*m_N+j] = pointer;
        }
    }
    
    float prob = 0;
    int pointer = 0;
    for(int j=0; j<m_N; j++)
    {
        if (prob < delta[(number-1)*m_N+j])
        {
            prob = delta[(number-1)*m_N+j];
            pointer = j;
        }
    }
    
    
    for(int t = number-1; t>=0; t--)
    {
        (*hiddenStates)[t] = pointer;
        pointer = phi[t*m_N+pointer];
    }
    
}

void HMM::train(int *observableStates, int T)
{
    // compute alpha
    float *alpha = new float[T * m_N];
    computeAlpha(alpha, observableStates, T);
    
    // compute beta
    float *beta = new float[T * m_N];
    computeBeta(beta, observableStates, T);    

    float *xi = new float[T * m_N * m_N];
    float *gamma = new float[T * m_N];
    computeXi(xi, gamma, observableStates, T, alpha, beta);
    
    //float *gamma = new float[T * m_N];
    //computeGamma(gamma, observableStates, T, alpha, beta);
    
    // update pi
    for (int i=0; i<m_N; i++)
    {
        m_pi[i] = gamma[i];
    }
    
    // update A
    for(int i=0; i<m_N; i++)
    {
        float sumGama = 0.0;
        for(int t=0; t<T-1; t++)
        {
            sumGama += gamma[t*m_N+i];
        }
        
        for(int j=0; j<m_N; j++)
        {
            float sumXi = 0.0;
            for(int t=0; t<T-1; t++)
            {
                sumXi += xi[t*m_N*m_N+i*m_N+j];
            }
            m_A[i*m_N+j] = sumXi/sumGama;
        }
    }
    
    // update B
    for(int i=0; i<m_N; i++)
    {
        float sumGama = 0.0;
        for(int t=0; t<T; t++)
        {
            sumGama += gamma[t*m_N+i];
        }
        
		for(int k=0; k<m_M; k++)
		{
			float sumGamaK = 0.0;
			for(int t=0; t<T; t++)
			{
				if(observableStates[t]==k)
				{
					sumGamaK += gamma[t*m_N+i];
				}
			}
			m_B[i*m_N+k] = sumGamaK/sumGama;
		}
    }

	delete [] alpha;
	delete [] beta;
	delete [] xi;
}

void HMM::computeAlpha(float *&alpha, int *observableStates, int T)
{
    for (int i=0; i<m_N; i++) 
    {
        alpha[i]= m_pi[i]*m_B[m_M*i+observableStates[0]];
    }
    
    for (int t=1; t<T; t++) 
    {
        for (int j=0; j<m_N; j++)
        {
            float sum = 0;
            for (int i=0; i<m_N; i++) 
            {
                sum += alpha[(t-1)*m_N+i]*m_A[i*m_N+j]*m_B[j*m_M+observableStates[t]];
            }
            alpha[t*m_N+j] = sum;
        }
    }
}

void HMM::computeBeta(float *&beta, int* observableStates, int T)
{
    for (int i=0; i<m_N; i++) 
    {
        beta[(T-1)*m_N+i]= 1;
    }
    
    for (int t=T-2; t>=0; t--) 
    {
        for (int i=0; i<m_N; i++)
        {
            float sum = 0;
            for (int j=0; j<m_N; j++) 
            {
                sum += beta[(t+1)*m_N+j]*m_A[i*m_N+j]*m_B[j*m_M+observableStates[t+1]];
            }
            beta[t*m_N+i] = sum;
        }
    }
}

void HMM::computeXi(float *&Xi, float *&gamma, int* observableStates, int T, float *alpha, float *beta)
{
	for (int t=0; t<T-1; t++)
    {
        float *xiT = Xi+(t*m_N*m_N);
		float *gammaT = gamma + t*m_N;

        float sum = 0.0;
        for(int i=0; i<m_N; i++)
        {
		    for(int j=0; j<m_N; j++)
            {
                xiT[i*m_N+j] = alpha[t*m_N+i]*m_A[i*m_N+j]*beta[(t+1)*m_N+j]*m_B[j*m_N+observableStates[t+1]];
                sum += xiT[i*m_N+j];
            }
        }
        
        for(int i=0; i<m_N; i++)
        {
			gammaT[i] = 0.0;
            for(int j=0; j<m_N; j++)
            {
                xiT[i*m_N+j] /= sum;
				gammaT[i] += xiT[i*m_N+j];
            }
        }
    }
}

void HMM::computeGamma(float *&gamma, int* observableStates, int T, float *alpha, float *beta)
{
	for(int t=0; t<T; t++)
    {
        float *gammaT = gamma+t*m_N;
        float sum = 0.0;
        for (int i=0; i<m_N; i++)
        {
            gammaT[i] = alpha[t*m_N+i]*beta[t*m_N+i];
            sum += gammaT[i];
        }
        
        for(int i=0; i<m_N; i++)
        {
            gammaT[i] /= sum;
        }
    }
}














