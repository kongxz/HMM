//
//  main.cpp
//  HMM
//
//  Created by Xiangzhen Kong on 12-6-27.
//  Copyright (c) 2012年 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "HMM.h"

void output(int *a, int len)
{
	for(int i=0; i<len; i++)
    {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void testHMM()
{
	 // insert code here...
    HMM h(3,4);
    
    int ob[5] = {3, 1, 1, 1, 2};
	int ob2[5] = {1, 0, 3, 1, 0};
	int hidden[5] = {0, 2, 2, 2, 1};
	int *res = new int[5];
    
	for (int i=0; i<5; i++)
	{	
		std::cout << "Probability: " << h.evaluate(ob, 5) << std::endl;
		std::cout << "Probability: " << h.evaluate(ob2, 5) << std::endl;
		//h.decode(ob, 5, &res);
		//output(res, 5);

		std::cout << "After train....." << std::endl;
		h.train(ob, 5);
	}
}

#include "BaseHMM.h"
#include "DiscreteHMM.h"


int main(int argc, const char * argv[])
{
	//testHMM();
	Matrix<float> discretePDF(3, 4);

	ObservationPDInterface *observal = new DiscretePD(discretePDF);
	BaseHMM baseHMM(3, observal);

	

	int p;   
	std::cin >> p ;
    return 0;
}

