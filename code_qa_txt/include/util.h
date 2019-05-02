#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <vector>

void str_split(const std::string &s, char delim, std::vector<std::string> &result);

void str_replace(std::string& str, std::string p, std::string q);

inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}

#endif