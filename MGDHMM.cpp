#include "MGDHMM.h"


MGDHMM::MGDHMM(int n, int m)
	: m_N(n)
	, m_M(m)
{
	init();
}


MGDHMM::~MGDHMM(void)
{
}

void MGDHMM::init()
{
	 m_pi = new float[m_N];
    m_A = new float[m_N*m_N];
    m_B = new GaussDistrib[m_N*m_M];
    
    for(int i=0; i<m_N; i++)
    {
        m_pi[i] = 1.0/m_N;
        for(int j=0; j<m_N; j++)
        {
            m_A[i*m_N + j] = 1.0/m_N;
        }
    }
}

