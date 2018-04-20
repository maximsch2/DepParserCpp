#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <folly/String.h>
#include <folly/Conv.h>

#include <caffe2/core/tensor.h>

#include <eigen3/Eigen/Dense>
using namespace Eigen;

const std::string T_UNK = "<UNK>";

struct EmbeddingData
{
    std::unordered_map<std::string, int> tok2id;
    std::vector<std::string> id2tok;

    MatrixXf embeddings;
    caffe2::TensorCPU tensor;
    int embedding_size;
    int NULL_ID;

    int get_embedding_id(const std::string &token);
    EmbeddingData(std::vector<std::string> tokens, int size);
    void load_from_file(std::string filename);
    bool contains(const std::string &token) const { return tok2id.find(token) != end(tok2id); }
};

EmbeddingData::EmbeddingData(std::vector<std::string> tokens, int size) : 
    embedding_size(size), id2tok(tokens), embeddings(tokens.size() + 2, size)
{
    std::cout << "Creating embedding based on " << tokens.size() << " tokens" << std::endl;
    int id = 0;
    for (const auto &token : tokens)
    {
        tok2id[token] = id;
        id += 1;
    }
    id2tok.push_back(T_UNK);
    tok2id[T_UNK] = id;
    NULL_ID = id + 1;
    embeddings.setRandom(); // uniform [-1, 1]

    tensor.Resize(embeddings.rows(), embedding_size);
    tensor.ShareExternalPointer(&embeddings(0, 0));
}

int EmbeddingData::get_embedding_id(const std::string &token)
{
    if (contains(token))
    {
        return tok2id[token];
    }
    else
    {
        return tok2id[T_UNK];
    }
}

void EmbeddingData::load_from_file(std::string filename)
{
    std::ifstream infile(filename);
    std::string line;
    int loaded = 0;
    VectorXf data(embedding_size);
    while (std::getline(infile, line))
    {
        std::vector<std::string> vec, vec2;
        folly::split('\t', line, vec);
        auto token = vec[0];
        if (!contains(token))
            continue;
        auto id = tok2id[token];

        folly::split(' ', vec[1], vec2);
        assert(vec2.size() >= embedding_size);
        for (auto i = 0; i < embedding_size; i++)
        {
            embeddings(id, i) = folly::to<float>(vec2[i]);
        }
        loaded += 1;
    }
    std::cout << "Loaded " << loaded << " word embeddings";
}
