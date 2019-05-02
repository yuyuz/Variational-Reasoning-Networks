#include "util.h"

void str_split(const std::string &s, char delim, std::vector<std::string> &result)
{
    std::stringstream ss;
    ss.str(s);
    std::string item;
    result.clear();
    while (std::getline(ss, item, delim))
        result.push_back(item);
}

void str_replace(std::string& str, std::string p, std::string q)
{
    while (true)
    {
        auto idx = str.find(p);
        if (idx == std::string::npos)
            break;
        str.replace(idx, p.size(), q);
    }
}