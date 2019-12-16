#include <iostream> 
#include <algorithm> 
#include "circle_table.h"

using namespace std; 


int main(void){ 
    circle_table circles[] ={ circle_table(99, 84, 88, 438), circle_table(79, 62, 44, 283), circle_table(80, 94, 87, 111), circle_table(88, 82, 67,483), circle_table(92,64, 93, 238), circle_table(99, 84, 52, 41), circle_table(99,55,85, 888), circle_table(82,90, 47, 281), circle_table(90,89, 50,842) }; 
    

    sort(circles, circles + 9); 

    for(int i=0; i<9; i++){ 
        cout << circles[i].y_center <<" , "<<circles[i].x_center<<" , "<<circles[i].radius<<" , "<<circles[i].arc_len<<endl; 
    }
    

    


    return 0; 
}



