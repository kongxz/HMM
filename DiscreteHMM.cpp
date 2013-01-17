#include "DiscreteHMM.h"


DiscreteSymbol::DiscreteSymbol(int symbol)
	: m_symbol(symbol)
{
}


DiscreteSymbol::~DiscreteSymbol(void)
{
}


/////////////////////////////////////////////////////////////

DiscretePD::DiscretePD(Matrix<float> pdf)
{
}

DiscretePD::~DiscretePD()
{
}

float DiscretePD::observationProbability(int state, SymbolInterface* symbol)
{
	DiscreteSymbol *dis = static_cast<DiscreteSymbol*>(symbol);

	return m_pdf.get(state, dis->getSymbol());
}

void DiscretePD::updatePDF(int state, int symbol, float probability)
{
	m_pdf.set(state, symbol, probability);
}



