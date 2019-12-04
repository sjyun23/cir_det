
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stack>
using namespace std;
using namespace cv;


int main()
{
    stack<int> st;  // default 는 deque 컨테이너 사용
    //stack<int, vector<int>> st;  vector 컨테이너를 이용하여 stack 컨테이너 생성

    st.push(10);  st.push(20);    st.push(30);

    cout << st.top() << endl;    // 스택 제일 위의 요소 가져오기
    cout << st.top() << endl;

    st.pop();                    // 스택 제일 위의 요소 제거
    cout << st.top() << endl;
    st.pop();

    cout << st.top() << endl;
    st.pop();
    
    if (st.empty())                // 스택이 비었는지 확인
        cout << "stack에 데이터 없음" << endl;


 return 0;
}


