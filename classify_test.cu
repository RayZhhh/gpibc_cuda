//
// Created by Derek on 2022/11/16.
//
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "classifier.h"

using namespace std;

typedef vector<vector<float>> vvf;


void split(const std::string &s, std::vector<std::string> &v, const std::string &c) {
    v.clear();
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}


void append_img_to_vec(const string &filename, vector<vector<float>> &data, vector<int> &label_set, int label) {
    ifstream in(filename, ios::in);
    string line;
    while (getline(in, line)) {
        vector<string> strs;
        split(line, strs, " ");
        vector<float> temp;
        for (auto &str: strs) {
            temp.push_back(stof(str) / (float) 255);
        }
        data.push_back(temp);
        label_set.emplace_back(label);
    }
}


int main() {
    srand(time(0));

    vvf mnist;
    vecI label;
    append_img_to_vec("../uiuc/0_train.txt", mnist, label, 1);
    append_img_to_vec("../uiuc/1_train.txt", mnist, label, -1);
    printf("data_size: %d; data_width: %d\n", (int) mnist.size(), (int) mnist[0].size());
    printf("label_size: %d\n", (int) label.size());

    auto classifier = BinaryClassifier(mnist, label, 40, 100);
    classifier.eval_batch = 10;
    classifier.generations = 50;
    classifier.init();

    auto start_time = clock();
    classifier.train();
    cout << "training time: " << (clock() - start_time) / CLOCKS_PER_SEC << "s" << endl;

    return 0;
}