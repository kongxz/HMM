#ifndef __MGDHMM_H__
#define __MGDHMM_H__

// Gauss Distribution

struct GaussDistrib
{
	float mu[24];
	float sigma[24];
};


// class MGDHMM
// Mixture Gauss Densities Hidden Markov Model.
class MGDHMM
{
public:
	MGDHMM(int n, int m);
	~MGDHMM(void);

	 // get the probability of observale states.
    float evaluate(int *observableStates, int T);
    
    // find hidden states
    void decode(int *observableStates, int number, int **hiddenStates);
    
    // train the model.
    void train(int *observableStates, int number);
    

private:
	void init();

private:
	int m_N;            // hidden states number
	int m_M;            // mixtrue number
	float *m_pi;        // initial probability
	float *m_A;         // transition probability matrix
	GaussDistrib *m_B;  // Gauss Distribution
	float *m_C;         // mixtrue coefficient.
};

#endif // __MGDHMM_H__