#ifndef __OBSERVATION_DISTRIBUTION_INTERFACE_H__
#define __OBSERVATION_DISTRIBUTION_INTERFACE_H__

/*

	class ObservationPDInterface

	Interface for observation probability distribution.
*/

class ObservationPDInterface
{
public:
	virtual float observationProbability(int state, SymbolInterface* symbol) = 0;
};

#endif // __OBSERVATION_DISTRIBUTION_INTERFACE_H__
