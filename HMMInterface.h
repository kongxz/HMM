//
//  HMMInterface.h
//
//  Created by Xiangzhen Kong on 12-6-27.
//  Copyright (c) 2012Äê __MyCompanyName__. All rights reserved.
//

#include <vector>

#include "SymbolInterface.h"

#ifndef HMM_INTERFACE_H
#define HMM_INTERFACE_H

class HMMInterface
{
    // get the probability of observale states.
    virtual float evaluate(std::vector<SymbolInterface*> observation) = 0;
    
    // find hidden states
    virtual std::vector<int> decode(std::vector<SymbolInterface*> observation) = 0;
    
    // train the model.
    virtual void train(std::vector<SymbolInterface*> observation) = 0;
};


#endif // HMM_INTERFACE_H
