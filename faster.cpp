#include <iostream>

using namespace std;

float calculate_error(float threshold, int dist_size, float d1[], float d2[])
{

    float d_size = dist_size/2;
   // cout << d_size << endl;

    int incorrect = 0;
    for(int i = 0; i < d_size; i++)
    {

        if(d1[i] < threshold){
            incorrect++;
        }
    }
    for(int i = 0; i < d_size; i++)
    {
        if(d2[i] > threshold){

            incorrect++;
        }
    }
    float fidelity = incorrect / (d_size*2);
    //cout << fidelity << endl;
    return fidelity;
}

float advanced_calculate_error(float threshold, int dist_size, int num_points, float s1[][2600], float s2[][2600])
{
    int s_size = dist_size/2;
    float d_size = dist_size/2;
    float summation = 0;
    float d1[s_size];
    for(int i = 0; i < s_size; i++){
        summation = 0;
        for(int j = 0; j < num_points; j++){
            summation += s1[i][j];
        }
        d1[i] = summation;
    }
    float d2[s_size];
    for(int i = 0; i < s_size; i++){
        summation = 0;
        for(int j = 0; j < num_points; j++){
            summation += s2[i][j];
        }
        d2[i] = summation;
    }
    int incorrect = 0;
    for(int i = 0; i < s_size; i++)
    {

        if(d1[i] < threshold){
            incorrect++;
        }
    }
    for(int i = 0; i < s_size; i++)
    {
        if(d2[i] > threshold){

            incorrect++;
        }
    }
    float fidelity = incorrect / (d_size*2);
    return fidelity;
}

extern "C" {
    float Calculate_error(float threshold, int dist_size, float d1[], float d2[])
    {
        return calculate_error(threshold, dist_size, d1, d2);
    }


    float Advanced_calculate_error(float threshold, int dist_size, int num_points, float d1[][2600], float d2[][2600])
    {
        return advanced_calculate_error(threshold, dist_size, num_points, d1, d2);
    }
}
