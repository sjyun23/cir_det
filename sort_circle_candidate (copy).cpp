#include <iostream> 
#include <algorithm> 
using namespace std; 

class circle_table{ 
    public: int y_center; int x_center; int radius; int arc_len;

    circle_table(int y_center, int x_center, int radius, int arc_len){ 
        this->y_center = y_center; 
        this->x_center = x_center;
        this->radius = radius;
        this->arc_len = arc_len;
        } 

    bool operator < (circle_table &circle){ 
        if(this->y_center == circle.y_center && this->x_center == circle.x_center){ 
            return radius > circle.radius;
        }else if(this->x_center == circle.x_center){
            return x_center > circle.x_center;
        }else{ 
            return y_center > circle.y_center; 
        } 
    }    
};



int main(void){ 
    circle_table circles[] ={ circle_table(99, 84, 88, 438), circle_table(79, 62, 44, 283), circle_table(80, 94, 87, 111), circle_table(88, 82, 67,483), circle_table(92,64, 93, 238), circle_table(99, 84, 52, 41), circle_table(99,55,85, 888), circle_table(82,90, 47, 281), circle_table(90,89, 50,842) }; 
    
    sort(circles, circles + 9); 

    for(int i=0; i<9; i++){ 
        cout << circles[i].y_center <<" , "<<circles[i].x_center<<" , "<<circles[i].radius<<" , "<<circles[i].arc_len<<endl; 
    }
    
    return 0; 
}



