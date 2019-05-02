#include "var_sample.h"
#include "dataset.h"

namespace gnn
{

SampleVar::SampleVar(std::string _name) : Variable(_name), samples(nullptr)
{

}

EleType SampleVar::GetEleType()
{
	return EleType::UNKNOWN;
}

MatMode SampleVar::GetMode()
{
    return MatMode::cpu;
}

void SampleVar::SetRef(void* p)
{
    samples = static_cast<std::vector<Sample*>*>(p);
}

template<typename T>
VectorVar<T>::VectorVar(std::string _name) : Variable(_name)
{
    vec = new std::vector<T>();
}

template<typename T>
EleType VectorVar<T>::GetEleType()
{
	return EleType::UNKNOWN;
}

template<typename T>
MatMode VectorVar<T>::GetMode()
{
    return MatMode::cpu;
}

template<typename T>
void VectorVar<T>::SetRef(void* p)
{
    vec = static_cast<std::vector<T>*>(p);
}

template class VectorVar<int>;
template class VectorVar<Node*>;
}