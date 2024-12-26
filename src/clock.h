#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

class PerformanceClock
{
public:
    PerformanceClock() { QueryPerformanceFrequency(&system_frequency); }
    double get_time_in_seconds() const
    {
        LARGE_INTEGER current_time;
        QueryPerformanceCounter(&current_time);
        double elapsedTime = static_cast<double>(current_time.QuadPart - start_time.QuadPart) / system_frequency.QuadPart;
        return elapsedTime;
    }
    void reset()
    {
        QueryPerformanceCounter(&start_time);
    }
private:
    LARGE_INTEGER system_frequency;
    LARGE_INTEGER start_time;
};