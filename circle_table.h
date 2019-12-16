#ifndef circle_table_h
#define circle_table_h

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

#endif