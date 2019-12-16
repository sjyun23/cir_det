#ifndef circle_table_h
#define circle_table_h

using namespace std;

class circle_table{
public:
    int y_center; int x_center; int radius; int arc_len; 

    circle_table(int y_center, int x_center, int radius, int arc_len ):y_center(y_center), x_center(x_center),radius(radius), arc_len(arc_len){}

    bool operator<(circle_table circle) const{  
        if(this->y_center == circle.y_center && this->x_center == circle.x_center && this->radius == circle.radius  ){
            return arc_len> circle.arc_len;

        }else if(this->y_center == circle.y_center && this->x_center == circle.x_center){ 
            return radius > circle.radius;
        }else if(this->y_center == circle.y_center){
            return x_center > circle.x_center;
        }else{ 
            return y_center > circle.y_center; 
        } 
    }
};

void Print(vector<circle_table> &v){
    cout << "circle_table overloading : "<< v.size()<<endl ;
    int print_len =v.size();

    if (v.size() > 100){
        int print_len=100;
    }

    for(int i=0; i<print_len; i++){
        cout << "[" << v[i].y_center << ", " << v[i].x_center<<" , " << v[i].radius << ", " << v[i].arc_len << "]"<<endl;
    }
}

#endif
