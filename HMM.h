//
//  HMM.h
//  HMM
//
//  Created by Xiangzhen Kong on 12-6-27.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//

#ifndef HMM_HMM_h
#define HMM_HMM_h

class HMM
{
public:
    HMM(int hidden, int observable);
    ~HMM();
    
    // get the probability of observale states.
    float evaluate(int *observableStates, int T);
    
    // find hidden states
    void decode(int *observableStates, int number, int **hiddenStates);
    
    // train the model.
    void train(int *observableStates, int number);
    
    
private:
    
    void init();
    
	void computeAlpha(float *&alpha, int *observableStates, int T); // Alpha is T * N matrix.
	void computeBeta(float *&beta, int* observableStates, int T);  // Beta is T * N matrix.
	 // Xi is (T-1) * N * N matrix.
	void computeXi(float *&Xi, float *&gamma, int* observableStates, int T, float *alpha, float *beta);   
	void computeGamma(float *&gamma, int* observableStates, int T, float *alpha, float *beta); // Gamma is T * N marix.
    
    
    int m_N;    // Hidden states number.
    int m_M;    // Observable states number.
    float *m_pi; // initial probability
    float *m_A;  // transition probability matrix
    float *m_B;  // confusion matrix
};


#endif
