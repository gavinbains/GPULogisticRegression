#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

void loadCSV(float** data, char* filename, int n_rows, int n_features) {
    ifstream file(filename);

    for(int row = 0; row < n_rows; row++)
    {
        string line;
        getline(file, line);
        if ( !file.good() )
            break;

        stringstream ss(line);
        float val;
        char comma;
        ss >> val; // skip row index
        for (int col = 0; col < n_features; col++)
        {
            ss >> comma;
            ss >> val;

            data[row][col] = val;

        }
    }
}

void normalize_data(float** data, int n_rows, int n_features) {
    for (int col = 1; col < n_features; col++) { // column-wise (skip first column which contains padding)

        int min = 0, max = 10; // calculate min and max
        for (int row = 0; row < n_rows; row++) {
            if (data[row][col] < min) {
                min = data[row][col];
            }
            if (data[row][col] > max) {
                max = data[row][col];
            }
        }

        int rng = max - min; // calculate range
        if (rng != 0) { // otherwise leave row as is
            for (int row = 0; row < n_rows; row++) {
                data[row][col] = 1 - ((max - data[row][col]) / rng); // normalize
            }
        }

    }
}

void relabel_yvec (int* yvec, int n_rows, int modelID, int n_models) {
    float high = 365.0 / n_models * modelID + 1;
    float low = 365.0 / n_models * (modelID-1);
    for (int i = 0; i < n_rows; i++) {
        int val = yvec[i];
        if (val >= low && val < high) {
            val = 1; // patient relapsed in this period
        } else {
            val = 0; // patient DID NOT relapse in this period
        }
        yvec[i] = val;
    }
}

void extract_yvec(float** data, int* yvec, int n_rows) {
    for (int i = 0; i < n_rows; i++) {
        yvec[i] = data[i][0]; // extract predictor
        data[i][0] = 1; // pads data
    }
}

void initialize_betas(float* betas, int n_features) {
    for (int i = 0; i < n_features; i++) {
        betas[i] = 0.0;
    }
}

void logistic_func(float* log_func_v, float* betas, float** data, int n_rows, int n_features) {
    for (int i = 0; i < n_rows; i++) {
        float temp = 0;
        for (int j = 0; j < n_features; j++) {
            temp += betas[j] * data[i][j];
        }
        log_func_v[i] = -1.0 * temp;
    }
    for (int i = 0; i < n_rows; i++) {
        log_func_v[i] = 1.0 / (1.0 + exp(log_func_v[i]));
    }
}

void log_gradient(float* gradient, float* betas, float** data, int* yvec, int n_rows, int n_features) {
    float* log_func_v = new float[n_rows];
    logistic_func(log_func_v, betas, data, n_rows, n_features);

    for (int i = 0; i < n_rows; i++) { // first_calc
        log_func_v[i] -= yvec[i];
    }

    for (int j = 0; j < n_features; j++) { // final_calc
        float temp = 0.0;
        for (int i = 0; i < n_rows; i++) {
            temp += log_func_v[i] * data[i][j];
        }
        gradient[j] = temp;
    }
}

float cost_func(float* betas, float** data, int* yvec, int n_rows, int n_features) {
    float* log_func_v = new float[n_rows];
    logistic_func(log_func_v, betas, data, n_rows, n_features);

    float total = 0.0;
    for (int i = 0; i < n_rows; i++) {
        float step1 = yvec[i] * log(log_func_v[i]);
        float step2 = (1 - yvec[i]) * log(1 - log_func_v[i]);
        total += -step1 - step2;
    }
    return total;
}

float grad_desc(float* betas, float** data, int* yvec, float lr, int max_iters, int n_rows, int n_features) {
    int num_iter = 1;
    for (int i = 0; i < max_iters; i++) {
        float* gradient = new float[n_features];
        log_gradient(gradient, betas, data, yvec, n_rows, n_features);
        for (int b = 0; b < n_features; b++) { // adjust betas
            betas[b] -= lr * gradient[b];
        }
        num_iter++;
    }
    float cost = cost_func(betas, data, yvec, n_rows, n_features);
    return cost;
}

float predict(float* betas, float** data, int* yvec, int n_rows, int n_features) {
    float* log_func_v = new float[n_rows];
    logistic_func(log_func_v, betas, data, n_rows, n_features);
    float threshold_step = 0.05;

    float optimal_percent_correct = 0;
    for (int t = 1; t <= 20; t++) {
        float threshold = threshold_step * t;
        float correct = 0.0;
        for (int i = 0; i < n_rows; i++) {
            if ((yvec[i] == 0 && log_func_v[i] <= threshold) || (yvec[i] == 1 && log_func_v[i] > threshold)) {
                correct++;
            }
        }
        float percent_correct = correct / n_rows * 100;
        if (percent_correct > optimal_percent_correct) {
            optimal_percent_correct = percent_correct;
        }
    }

    return optimal_percent_correct;
}

int main() {
    int n_rows = 100, n_features = 26;
    int n_models = 12, modelID = 4; // model predicting likelihood of relapse in month 4
    int max_iters = 1000000;
    float lr = 0.01;

    cout << "--- Loading training data...";
    char* training_filename = (char*)"training_data.csv";
    float** train = new float*[n_rows]; // allocate memory for data
    for (int i = 0; i < n_rows; i++) {
        train[i] = new float[n_features];
    }
    loadCSV(train, training_filename, n_rows, n_features+1);
    cout << " done! ---" << endl << endl;

    cout << "--- Loading testing data...";
    char* testing_filename = (char*)"testing_data.csv";
    float** test = new float*[n_rows]; // allocate memory for data
    for (int i = 0; i < n_rows; i++) {
        test[i] = new float[n_features];
    }
    loadCSV(test, testing_filename, n_rows, n_features);
    cout << " done! ---" << endl << endl;

    cout << "--- Extracting and re-labelling predictor...";
    int* yvec = new int[n_rows];
    extract_yvec(train, yvec, n_rows);
    // relabel_yvec(yvec, n_rows, modelID, n_models);
    cout << " done! ---" << endl << endl;

    cout << "--- Training Model..." << flush;
    float* betas = new float[n_features];
    initialize_betas(betas, n_features);
    float optimal_cost = grad_desc(betas, train, yvec, lr, max_iters, n_rows, n_features);
    cout << " done! ---" << endl << endl;

    cout << "Cost at solution: " << optimal_cost << endl;
    cout << "Learned Betas: ";
    for (int i = 0; i < n_features; i++) {
        cout << betas[i] << ' ';
    }
    cout << endl << endl;

    cout << "--- Measuring Performance...";
    float percent_correct = predict(betas, test, yvec, n_rows, n_features);
    cout << endl;
    cout << " done! ---" << endl << endl;
    cout << "Percent correct: " << percent_correct << "%" << endl;

    for (int i = 0; i < n_rows; i++) {
        delete train[i];
        delete test[i];
    }
    delete [] train;
    delete [] test;
}
