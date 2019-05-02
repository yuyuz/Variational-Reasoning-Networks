#ifndef VAR_SAMPLE_H
#define VAR_SAMPLE_H

#include "nn/variable.h"
#include "dataset.h"

namespace gnn
{

class SampleVar : public Variable
{
public:
	SampleVar(std::string _name);

	virtual EleType GetEleType() override;

	virtual MatMode GetMode() override;

	virtual void SetRef(void* p) override;	

	std::vector<Sample*>* samples;
};

template<typename T>
class VectorVar : public Variable
{
public:
	VectorVar(std::string _name);

	virtual EleType GetEleType() override;
	
	virtual MatMode GetMode() override;

	virtual void SetRef(void* p) override;	

	std::vector<T>* vec;
};

}

#endif