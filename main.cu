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
            temp.push_back(stof(str));
        }
        data.push_back(temp);
        label_set.emplace_back(label);
    }
}

#define RUNNING_TIMES 10

void run_test(const string& test_name, int h, int w, int eval_batch=10, const string& res_path = "res.csv") {
    ofstream out(res_path, ios::app);

    // write test name
    out << test_name << endl;

    // load data and run test
    vvf train_data;
    vecI train_label;
    append_img_to_vec(test_name + "/0_train.txt", train_data, train_label, 1);
    append_img_to_vec(test_name + "/1_train.txt", train_data, train_label, -1);

    printf("data_size: %d; data_width: %d\n", (int) train_data.size(), (int) train_data[0].size());
    printf("label_size: %d\n", (int) train_label.size());

    vvf test_data;
    vecI test_label;
    append_img_to_vec(test_name + "/0_test.txt", test_data, test_label, 1);
    append_img_to_vec(test_name + "/1_test.txt", test_data, test_label, -1);
    printf("test_data_size: %d; test_data_width: %d\n", (int) test_data.size(), (int) test_data[0].size());
    printf("test_label_size: %d\n", (int) test_label.size());

    for (int i = 0; i < RUNNING_TIMES; i++) {
        auto classifier = BinaryClassifier(train_data, train_label, test_data, test_label, h, w);
        classifier.eval_batch = eval_batch;
        classifier.generations = 50;
        classifier.init();

        // train
        auto start_time = clock();
        classifier.train();
        auto dur = (float) (clock() - start_time) / (float) CLOCKS_PER_SEC;
        cout << "training time: " << dur << "s" << endl;

        // test
        classifier.run_test();
        cout << endl;

        // write training time and test accuracy
        out << dur << "," << classifier.best_test_program.fitness << endl;
    }
}


int main(int argc, const char *argv[]) {
    auto fname = string(argv[1]);
    auto h = stoi(argv[2]);
    auto w = stoi(argv[3]);
    auto eval_batch = stoi(argv[4]);
    string res_path;
    if (argv[5] != nullptr) {
        res_path = string(argv[5]);
    } else {
        res_path = "res.csv";
    }

    run_test(fname, h, w, eval_batch, res_path);

    return 0;
}