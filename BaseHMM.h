#ifndef __BASE_HMM_H__
#define __BASE_HMM_H__

#include "HMMInterface.h"
#include "ObservationPDInterface.h"
#include "SymbolInterface.h"
#include "Matrix.h"


class BaseHMM : public HMMInterface
{
public:
	BaseHMM(int stateNum, ObservationPDInterface *b);
	~BaseHMM(void);

	 // get the probability of observale states.
    float evaluate(std::vector<SymbolInterface*> observation);
    
    // find hidden states
    std::vector<int> decode(std::vector<SymbolInterface*> observation);
    
    // train the model.
    void train(std::vector<SymbolInterface*> observation);
    
    
private:
    void init();
    
	//void computeAlpha(float *&alpha, int *observableStates, int T); // Alpha is T * N matrix.
	//void computeBeta(float *&beta, int* observableStates, int T);  // Beta is T * N matrix.
	// // Xi is (T-1) * N * N matrix.
	//void computeXi(float *&Xi, float *&gamma, int* observableStates, int T, float *alpha, float *beta);   
	//void computeGamma(float *&gamma, int* observableStates, int T, float *alpha, float *beta); // Gamma is T * N marix.
 //   
	// Alpha is T * N matrix.
	Matrix<float> computeAlpha(std::vector<SymbolInterface*> &observation);

	// Beta is T * N matrix.
	Matrix<float> computeBeta(std::vector<SymbolInterface*> &observation);

	// Gamma is T * N marix.
	Matrix<float> computeGamma(std::vector<SymbolInterface*> &observation, Matrix<float> &alpha, Matrix<float> &beta);

	// Xi is (T-1) * N * N matrix.
	std::vector<Matrix<float>> computeXi(std::vector<SymbolInterface*> &observation, Matrix<float> &gamma, Matrix<float> &alpha, Matrix<float> &beta);
    
    int m_N;    // Hidden states number.
	std::vector<float> m_pi; // initial probability
    Matrix<float> m_A;  // transition probability matrix
	ObservationPDInterface *m_B;
};

#endif // __BASE_HMM_H__
