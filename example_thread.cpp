
//// Online C++ compiler to run C++ program online
//#include <iostream>
//#include <thread>
//#include <queue>
//
//using namespace std;
//
//queue<int> myQueue;
//
//
//std::thread main1;
//std::thread second;
//void add() {
//
//
//    main1 = std::thread([&]() {
//        // main([&]()
//        int i = 0;
//        while (true) {
//            myQueue.push(i);
//            i++;
//            std::cout << "Adding  : " << i << std::endl;
//            if (i == 100) {
//                i = 0;
//                //break;
//            }
//        }
//    });
//
//    main1.join();
//
//    
//}
//
//void subtract() {
//
//    second = std::thread([&]() {
//        while (true) {
//            if (!myQueue.empty()) {
//                int element = myQueue.front();
//                std::cout << "Removing  : " << element << std::endl;
//                myQueue.pop();
//            }
//        }
//    });
//    second.join();
//    
//}
//
//
//
//int main() {
//    // Write C++ code here
//    std::cout << "Try programiz.pro" << std::endl;
//    add();
//    subtract();
//
//    /*main1.join();
//    second.join();*/
//    return 0;
//}



// Online C++ compiler to run C++ program online
#include <iostream>
#include <thread>
#include <queue>
#include <chrono>

using namespace std;

queue<int> myQueue;


std::thread main1;
std::thread second;
void add() {


    main1 = std::thread([&]() {
        // main([&]()
        int i = 0;
        while (true) {
            myQueue.push(i);
            i++;
            std::cout << "Adding  : " << i << std::endl;
            if (i == 100) {
                i = 0;
                //break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        });

   

}

void subtract() {

    second = std::thread([&]() {
        while (true) {
            if (!myQueue.empty()) {
                int element = myQueue.front();
                std::cout << "Removing  : " << element << std::endl;
                myQueue.pop();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        }
    });
    

}



int main() {
    // Write C++ code here
    std::cout << "Try programiz.pro" << std::endl;
    add();
    subtract();

    main1.join();
    second.join();
    return 0;
}
