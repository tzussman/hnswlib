#include "../../hnswlib/hnswlib.h"


int main() {
    int dim = 128;               // Dimension of the elements
    int max_elements = 100000;   // Maximum number of elements, should be known beforehand
    int M = 32;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(42);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    time_t start = clock();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }
    time_t end = clock();
    double add_time = (double) (end - start) / CLOCKS_PER_SEC;

    // Query the elements for themselves and measure recall
    float correct = 0;
    long total_time = 0;

    for (int i = 0; i < max_elements; i++) {
        time_t start = clock();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        time_t end = clock();
        total_time += end - start;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    float avg_time = (float)(total_time / max_elements) / CLOCKS_PER_SEC;
    std::cout << "Add time: " << add_time << std::endl;
    std::cout << "Avg search time: " << avg_time << std::endl;

    // Serialize index
    std::string hnsw_path = "hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;
    std::cout << "Recall of deserialized index: " << recall << "\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}
