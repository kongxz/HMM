#ifndef __DISCRETE_HMM_H__
#define __DISCRETE_HMM_H__

#include "SymbolInterface.h"
#include "ObservationPDInterface.h"
#include "Matrix.h"

class DiscreteSymbol : public SymbolInterface
{
public:
	DiscreteSymbol(int symbol);
	~DiscreteSymbol(void);

	int getSymbol()
	{
		return m_symbol;
	}

private:
	int m_symbol;
};

class DiscretePD : public ObservationPDInterface
{
public:
	DiscretePD(Matrix<float> pdf);
	~DiscretePD();

	float observationProbability(int state, SymbolInterface* symbol);

	void updatePDF(int state, int symbol, float probability);

private:
	Matrix<float> m_pdf;
};

#endif //__DISCRETE_HMM_H__